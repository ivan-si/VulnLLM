import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModel
)
from datasets import load_dataset
import gc
import os
import argparse
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import logging
from pynvml import *
import random
from datetime import datetime
import sys
import json

# Add argument parser with more detailed help messages
parser = argparse.ArgumentParser(
    description='Train a vulnerability detection model using LLM fine-tuning',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Train model with default settings
  python train_sm.py
  
  # Train with debug information (memory usage, etc.)
  python train_sm.py --debug
  
  # Save sample prompts to files for inspection
  python train_sm.py --sample-prompts
  
  # Run with both debugging and prompt sampling
  python train_sm.py --debug --sample-prompts
""")

parser.add_argument(
    '--debug',
    action='store_true',
    help='Enable debug output including detailed GPU memory tracking and tensor allocation information'
)

parser.add_argument(
    '--sample-prompts',
    action='store_true',
    help='Log sample prompts to files (sample_prompts_*.txt) to verify prompt formatting and data loading'
)

args = parser.parse_args()

# Setup logging with debug level if requested
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables and device
# Adjusted for memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clear_memory():
    """Helper function to clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_program_name() -> str:
    """Get the name of the currently running program."""
    return os.path.basename(sys.argv[0])


def extract_model_name(model_path: str) -> str:
    """Extract the model name from its loading path."""
    # Remove any potential organization prefix (e.g., 'internlm/' from 'internlm/internlm2_5-1_8b-chat')
    return model_path.split('/')[-1]


def get_prompt_template() -> str:
    """Get the standardized prompt template."""
    return (
        "You are an expert in code security. Below is a code snippet. "
        "Analyze it and decide if there is a vulnerability:\n\n"
        "```\n{item[self.text_column]}\n```\n\n"
        "Is this code vulnerable? Respond only with '1' for vulnerable or '0' for not vulnerable."
    )


def log_evaluation_metrics(
    eval_results: dict,
    epoch: int,
    program_name: str = None,
    model_name: str = None,
    csv_path: str = "evaluation_results.csv",
    code_text: str = get_prompt_template()
) -> None:
    """
    Log evaluation metrics to a CSV file, creating it if it doesn't exist.

    Args:
        eval_results: Dictionary containing evaluation metrics
        epoch: Current training epoch
        program_name: Name of the program/experiment (optional, auto-detected if None)
        model_name: Name of the LLM model used (optional, should be set during trainer initialization)
        csv_path: Path to the CSV file
    """

    # Auto-detect program name if not provided
    if program_name is None:
        program_name = get_program_name()

    # Prepare the metrics data
    metrics_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'program_name': program_name,
        'model_name': model_name,
        'epoch': epoch,
        'prompt_template': get_prompt_template(),
        'accuracy': eval_results.get('eval_accuracy', None),
        'balanced_accuracy': eval_results.get('eval_balanced_accuracy', None),
        'precision': eval_results.get('eval_precision', None),
        'recall': eval_results.get('eval_recall', None),
        'f1': eval_results.get('eval_f1', None),
        'combined_metric': eval_results.get('eval_combined_metric', None),
        'loss': eval_results.get('eval_loss', None),
        'code_text': get_prompt_template()
    }

    # Convert to DataFrame
    results_df = pd.DataFrame([metrics_data])

    # Check if file exists and append or create accordingly
    if os.path.exists(csv_path):
        results_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(csv_path, index=False)


def perform_initial_evaluation(trainer, output_dir: str = "./vulnerability_detector"):
    """
    Perform initial evaluation before training and log results to CSV.

    Args:
        trainer: The MemoryTrackingTrainer instance
        output_dir: Directory where results will be saved
    """
    logger.info("Performing initial evaluation before training...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "initial_evaluation.csv")

    # Perform evaluation
    initial_results = trainer.evaluate()

    # Prepare results dictionary with all necessary fields
    results_dict = {
        'stage': 'pre_training',
        'epoch': 0,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': trainer.model_name,
        'program_name': trainer.program_name,
        'eval_accuracy': initial_results.get('eval_accuracy', None),
        'eval_balanced_accuracy': initial_results.get('eval_balanced_accuracy', None),
        'eval_precision': initial_results.get('eval_precision', None),
        'eval_recall': initial_results.get('eval_recall', None),
        'eval_f1': initial_results.get('eval_f1', None),
        'eval_combined_metric': initial_results.get('eval_combined_metric', None),
        'eval_loss': initial_results.get('eval_loss', None)
    }

    # Convert to DataFrame
    results_df = pd.DataFrame([results_dict])

    try:
        # Save to CSV with explicit encoding
        results_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(
            f"Successfully saved initial evaluation results to {csv_path}")

        # Verify file was created
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            logger.info(
                f"CSV file created successfully. Size: {file_size} bytes")

            # Read back and log first few lines for verification
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_lines = ''.join(f.readlines()[:5])
                logger.info(f"First few lines of CSV:\n{first_lines}")
        else:
            logger.error(f"CSV file was not created at {csv_path}")

    except Exception as e:
        logger.error(f"Error saving CSV file: {str(e)}")
        # Try alternative location
        backup_path = "initial_evaluation.csv"
        logger.info(
            f"Attempting to save to alternative location: {backup_path}")
        results_df.to_csv(backup_path, index=False, encoding='utf-8')

    return initial_results


def get_tensor_memory_usage():
    """Print summarized memory usage of all tensors"""
    if not args.debug:
        return

    print("\n=== Tensor Memory Summary ===")
    device_totals = {}
    dtype_totals = {}
    total_size = 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                size = obj.nelement() * obj.element_size()
                total_size += size

                # Aggregate by device
                device = str(obj.device)
                device_totals[device] = device_totals.get(device, 0) + size

                # Aggregate by dtype
                dtype = str(obj.dtype)
                dtype_totals[dtype] = dtype_totals.get(dtype, 0) + size
        except:
            pass

    print("\nBy Device:")
    for device, size in device_totals.items():
        print(f"{device}: {size / 1024**3:.2f} GB")

    print("\nBy Data Type:")
    for dtype, size in dtype_totals.items():
        print(f"{dtype}: {size / 1024**3:.2f} GB")

    print(f"\nTotal Tensor Memory: {total_size / 1024**3:.2f} GB")


def get_gpu_memory_map():
    """Get a summary of GPU memory usage"""
    try:
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        summary = {}

        total_used = 0
        total_free = 0
        total_capacity = 0

        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)

            total_used += info.used
            total_free += info.free
            total_capacity += info.total

        summary = {
            'total_capacity_gb': total_capacity / 1024**3,
            'total_used_gb': total_used / 1024**3,
            'total_free_gb': total_free / 1024**3,
            'utilization_percent': (total_used / total_capacity) * 100
        }

        return summary
    except NVMLError as error:
        logger.error(f"Error getting GPU memory map: {error}")
        return {}
    finally:
        try:
            nvmlShutdown()
        except:
            pass


def print_gpu_memory_stats(message: Optional[str] = None):
    """Print GPU memory statistics"""
    if not args.debug:
        return

    if message:
        print(f"\n=== GPU Memory Stats: {message} ===")
    else:
        print("\n=== GPU Memory Stats ===")

    # PyTorch memory stats
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"PyTorch Memory (GB) - Allocated: {allocated:.2f}, Reserved: {reserved:.2f}, Max: {max_allocated:.2f}")

    # NVIDIA-SMI summary
    gpu_summary = get_gpu_memory_map()
    if gpu_summary:
        print(f"GPU Utilization: {gpu_summary['utilization_percent']:.1f}% "
              f"({gpu_summary['total_used_gb']:.2f}GB used / {gpu_summary['total_capacity_gb']:.2f}GB total)")


def log_sample_prompts(dataset, num_samples: int = 5, output_file: str = "sample_prompts.txt"):
    """Log a random sample of prompts from the dataset to a file"""
    if not args.sample_prompts:
        return

    indices = random.sample(range(len(dataset)),
                            min(num_samples, len(dataset)))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Sample Prompts from Dataset (n={num_samples}) ===\n\n")

        for idx in indices:
            item = dataset.dataset[idx]
            code_text = (
                f"You are an expert in code security. Below is a code snippet. "
                f"Analyze it and decide if there is a vulnerability:\n\n"
                f"```\n{item[dataset.text_column]}\n```\n\n"
                f"Is this code vulnerable? Respond only with '1' for vulnerable or '0' for not vulnerable."
            )

            f.write(f"=== Sample {idx} ===\n")
            f.write(f"Original target: {item['target']}\n")
            f.write("Generated prompt:\n")
            f.write(code_text)
            f.write("\n\n" + "="*50 + "\n\n")


def print_model_memory_usage(model):
    """Print memory usage of model parameters"""
    print("\n=== Model Parameter Memory Usage ===")
    total_params = 0
    total_size = 0

    for name, param in model.named_parameters():
        param_size = param.nelement() * param.element_size()
        total_params += param.nelement()
        total_size += param_size
        print(f"{name}: Size: {param.size()}, "
              f"Type: {param.dtype}, "
              f"Memory: {param_size / 1024**2:.2f} MB")

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total Size: {total_size / 1024**3:.2f} GB")


class VulnerabilityModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = self.base_model.config.hidden_size

        # Classifier head remains the same
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.LayerNorm(hidden_size // 4),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size // 4, 1),
        )

        # Enable gradient checkpointing
        self.base_model.config.use_cache = False
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Required method for compatibility with Trainer"""
        self.base_model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)
        return logits


class VulnerabilityDataset(Dataset):
    def __init__(
        self,
        dataset: Dict,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        text_column: str = "normalized_func",
        num_samples: int = 5
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

        # Log sample prompts if requested
        log_file = f"sample_prompts_{id(self)}.txt"
        log_sample_prompts(self, num_samples, log_file)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Rest of the implementation remains the same...
        item = self.dataset[idx]

        code_text = (
            f"You are an expert in code security. Below is a code snippet. "
            f"Analyze it and decide if there is a vulnerability:\n\n"
            f"```\n{item[self.text_column]}\n```\n\n"
            f"Is this code vulnerable? Respond only with '1' for vulnerable or '0' for not vulnerable."
        )

        encodings = self.tokenizer(
            code_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_token_type_ids=False
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(item['target'], dtype=torch.float)
        }


def calculate_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate metrics using weighted calculations to account for class imbalance"""
    binary_preds = (preds > 0).astype(np.int64)

    # Convert to PyTorch tensors
    preds_tensor = torch.tensor(binary_preds)
    labels_tensor = torch.tensor(labels)

    # Calculate base metrics
    tp = torch.sum((preds_tensor == 1) & (labels_tensor == 1)).float()
    fp = torch.sum((preds_tensor == 1) & (labels_tensor == 0)).float()
    tn = torch.sum((preds_tensor == 0) & (labels_tensor == 0)).float()
    fn = torch.sum((preds_tensor == 0) & (labels_tensor == 1)).float()

    # Calculate metrics with class weights
    # [negative_weight, positive_weight]
    weights = torch.tensor([0.9232, 1.0907])

    # Weighted accuracy
    weighted_correct = (tn * weights[0] + tp * weights[1])
    weighted_total = ((tn + fp) * weights[0] + (tp + fn) * weights[1])
    weighted_accuracy = weighted_correct / weighted_total

    # Standard metrics
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    # Calculate balanced accuracy
    balanced_acc = (recall + tn / (tn + fp + 1e-7)) / 2

    # Calculate combined metric (average of F1 and balanced accuracy)
    combined_metric = (f1 + balanced_acc) / 2

    return {
        'accuracy': weighted_accuracy.item(),
        'balanced_accuracy': balanced_acc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'combined_metric': combined_metric.item()  # New combined metric
    }


def compute_metrics(pred):
    """Compute metrics for the model evaluation"""
    return calculate_metrics(pred.predictions, pred.label_ids)


class VulnerabilityTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        clear_memory()

        # Threshold for binary classification
        self.threshold = 0.5

        # Set pos_weight based on class weight calculation
        self.pos_weight = torch.tensor([1.1815])

        # Store class weights for potential use in weighted metrics
        self.class_weights = {
            0: 0.9232,  # weight for negative class
            1: 1.0907   # weight for positive class
        }

        # Store best metrics
        self.best_f1 = 0.0
        self.best_balanced_acc = 0.0

    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels').to(self.args.device)
        input_ids = inputs['input_ids'].to(self.args.device)
        attention_mask = inputs['attention_mask'].to(self.args.device)

        logits = model(input_ids, attention_mask)
        logits = logits.view(-1)
        labels = labels.float()

        # Use BCEWithLogitsLoss with our calculated pos_weight
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(self.args.device)
        )
        loss = loss_fct(logits, labels)

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            labels = inputs.pop('labels').to(self.args.device)
            input_ids = inputs['input_ids'].to(self.args.device)
            attention_mask = inputs['attention_mask'].to(self.args.device)

            logits = model(input_ids, attention_mask)
            logits = logits.view(-1)

            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()

            loss = None
            if labels is not None:
                labels = labels.float()
                loss_fct = torch.nn.BCEWithLogitsLoss(
                    pos_weight=self.pos_weight.to(self.args.device)
                )
                loss = loss_fct(logits, labels)

            return (loss, preds, labels)

    def log_metrics(self, split, metrics):
        """Custom metric logging to track both F1 and balanced accuracy"""
        super().log_metrics(split, metrics)

        if split == "eval":
            current_f1 = metrics.get("eval_f1", 0.0)
            current_balanced_acc = metrics.get("eval_balanced_accuracy", 0.0)

            # Update best scores
            self.best_f1 = max(self.best_f1, current_f1)
            self.best_balanced_acc = max(
                self.best_balanced_acc, current_balanced_acc)

            # Log best scores
            self.log({
                "best_f1": self.best_f1,
                "best_balanced_accuracy": self.best_balanced_acc
            })


class MemoryTrackingTrainer(VulnerabilityTrainer):
    def __init__(self, *args, model_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.program_name = get_program_name()
        self.model_name = extract_model_name(
            model_path) if model_path else None
        self.debug_enabled = kwargs.get('debug_enabled', False)
        self.debug_print_interval = kwargs.get('debug_print_interval', 100)
        self.step_counter = 0
        self.current_epoch = 0
        self.last_save_path = None

        # Initialize training metrics tracking
        self.training_metrics = {
            'epoch_losses': [],
            'learning_rates': [],
            'grad_norms': [],
            'memory_usage': []
        }

        # Setup additional logging
        self.train_log_file = os.path.join(
            self.args.output_dir, 'detailed_training_log.jsonl')
        os.makedirs(self.args.output_dir, exist_ok=True)

    def evaluate(self, *args, **kwargs):
        try:
            if self.debug_enabled:
                print("\nEvaluation Phase")
                print_gpu_memory_stats("Before evaluation")
        except Exception as e:
            print(f"Debug logging failed during evaluation: {str(e)}")

        results = super().evaluate(*args, **kwargs)

        # Get the prompt template from the evaluation dataset
        code_text = None
        if hasattr(self.eval_dataset, 'text_column') and len(self.eval_dataset.dataset) > 0:
            code_text = self.eval_dataset.dataset[0][self.eval_dataset.text_column]

        # Log the evaluation results
        log_evaluation_metrics(
            eval_results=results,
            epoch=self.current_epoch,
            code_text=get_prompt_template(),
            program_name=self.program_name,
            model_name=self.model_name
        )

        try:
            if self.debug_enabled:
                print_gpu_memory_stats("After evaluation")
                print("-" * 50)
        except Exception as e:
            print(f"Debug logging failed during evaluation: {str(e)}")

        return results

    def log_training_metrics(self, loss, learning_rate, grad_norm):
        """Log detailed training metrics to a JSONL file"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'epoch': self.current_epoch,
            'step': self.step_counter,
            'loss': float(loss),
            'learning_rate': float(learning_rate),
            'gradient_norm': float(grad_norm),
            'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }

        # Append to JSONL file
        with open(self.train_log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

        # Update tracking dictionaries
        self.training_metrics['epoch_losses'].append(loss)
        self.training_metrics['learning_rates'].append(learning_rate)
        self.training_metrics['grad_norms'].append(grad_norm)
        self.training_metrics['memory_usage'].append(
            metrics['memory_allocated_gb'])

    def training_step(self, model, inputs):
        """Enhanced training step with better epoch tracking and detailed logging"""
        # Update epoch counter based on current step and total steps
        steps_per_epoch = len(
            self.train_dataset) // (self.args.train_batch_size * self.args.gradient_accumulation_steps)
        self.current_epoch = self.step_counter // steps_per_epoch

        # Debug logging at interval
        if self.debug_enabled and (self.step_counter % self.debug_print_interval == 0):
            logger.debug(
                f"\nEpoch {self.current_epoch}, Step {self.step_counter}")
            print_gpu_memory_stats(f"Step {self.step_counter} (Before)")

        try:
            # Compute loss and backward pass
            loss = super().training_step(model, inputs)

            # Get learning rate and gradient norm
            current_lr = self.optimizer.param_groups[0]['lr']
            grad_norm = self.get_gradient_norm()

            # Log metrics
            self.log_training_metrics(loss.item(), current_lr, grad_norm)

            # Performance monitoring
            if self.debug_enabled and (self.step_counter % self.debug_print_interval == 0):
                print_gpu_memory_stats(f"Step {self.step_counter} (After)")
                logger.debug(
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Grad Norm: {grad_norm:.2f}"
                )

            self.step_counter += 1
            return loss

        except Exception as e:
            logger.error(
                f"Error in training step {self.step_counter}: {str(e)}")
            print_gpu_memory_stats("At error")
            raise

    def get_gradient_norm(self):
        """Calculate total gradient norm"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def on_save(self, args, state, control, **kwargs):
        """Track model saves and update metrics"""
        if state.best_model_checkpoint is not None:
            self.last_save_path = state.best_model_checkpoint

            # Save training progress visualization
            self.plot_training_progress()

        return super().on_save(args, state, control, **kwargs)

    def plot_training_progress(self):
        """Generate and save training progress visualizations"""
        try:
            import matplotlib.pyplot as plt

            # Create directory for plots
            plots_dir = os.path.join(self.args.output_dir, 'training_plots')
            os.makedirs(plots_dir, exist_ok=True)

            # Plot loss
            plt.figure(figsize=(10, 6))
            plt.plot(self.training_metrics['epoch_losses'])
            plt.title('Training Loss Progress')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(plots_dir, 'loss_progress.png'))
            plt.close()

            # Plot learning rate
            plt.figure(figsize=(10, 6))
            plt.plot(self.training_metrics['learning_rates'])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.savefig(os.path.join(plots_dir, 'lr_schedule.png'))
            plt.close()

            # Plot memory usage
            plt.figure(figsize=(10, 6))
            plt.plot(self.training_metrics['memory_usage'])
            plt.title('GPU Memory Usage')
            plt.xlabel('Step')
            plt.ylabel('Memory (GB)')
            plt.savefig(os.path.join(plots_dir, 'memory_usage.png'))
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to generate training plots: {str(e)}")


def main():
    clear_memory()
    MODEL_PATH = "internlm/internlm2_5-1_8b-chat"
    print_gpu_memory_stats("After clear_memory")
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True)
    print_gpu_memory_stats("After loading tokenizer")

    base_model = model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True)
    print_gpu_memory_stats("After loading base model")

    # Create custom model with classification head
    model = VulnerabilityModel(base_model)
    model.to(DEVICE)
    model.gradient_checkpointing_enable()
    print_gpu_memory_stats("After moving model to GPU")

    logger.info("Loading and preparing datasets...")
    ds = load_dataset("DetectVul/devign")

    # Create datasets with logging enabled
    train_dataset = VulnerabilityDataset(
        ds['train'],
        tokenizer,
        max_length=256,
        num_samples=5
    )

    val_dataset = VulnerabilityDataset(
        ds['validation'],
        tokenizer,
        max_length=256,
        num_samples=3
    )

    # Create output directory
    output_dir = "./vulnerability_detector"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        eval_steps=100,
        save_total_limit=4,
        gradient_accumulation_steps=32,
        fp16=True,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.02,
        optim="adamw_torch",
        gradient_checkpointing=True,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="combined_metric",
        greater_is_better=True,
        report_to="tensorboard"
    )

    trainer = MemoryTrackingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        model_path=MODEL_PATH,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.01
            )
        ]
    )

    # Perform initial evaluation if in debug mode
    if args.debug:
        logger.info("Debug mode enabled - performing initial evaluation")
        initial_results = perform_initial_evaluation(
            trainer, output_dir=output_dir)

        # Log initial results to console as well
        logger.info("Initial Evaluation Results:")
        for key, value in initial_results.items():
            logger.info(f"{key}: {value}")

    logger.info("Starting training...")
    try:
        trainer.train()
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
        logger.info(f"Best F1 Score: {trainer.best_f1:.4f}")
        logger.info(f"Best Balanced Accuracy: {trainer.best_balanced_acc:.4f}")

        # Save the model with additional metric information
        final_output_dir = "./vulnerability_detector_final"
        os.makedirs(final_output_dir, exist_ok=True)
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)

        # Save final metrics
        metrics_path = os.path.join(final_output_dir, "final_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Best F1 Score: {trainer.best_f1:.4f}\n")
            f.write(
                f"Best Balanced Accuracy: {trainer.best_balanced_acc:.4f}\n")
            f.write(f"Final Metrics: {eval_results}\n")

            # Add initial evaluation results if they exist
            if args.debug:
                f.write("\nInitial Evaluation Metrics:\n")
                f.write(f"{initial_results}\n")

    except Exception as e:
        print_gpu_memory_stats("At error")
        get_tensor_memory_usage()
        print("Current GPU Memory Map:")
        print(get_gpu_memory_map())
        logger.error(f"An error occurred during training: {str(e)}")
        raise
    finally:
        clear_memory()


if __name__ == "__main__":
    main()
