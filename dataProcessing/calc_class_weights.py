import numpy as np
import pandas as pd
from datasets import Dataset
import logging

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def compute_balanced_class_weights(labels):
    """
    Compute balanced class weights manually.
    Weight = total_samples / (n_classes * samples_per_class)
    """
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    n_classes = len(unique_classes)
    
    # Calculate weights for each class
    weights = total_samples / (n_classes * class_counts)
    return dict(zip(unique_classes, weights))

def load_and_analyze_dataset():
    """Load the dataset and analyze class distribution"""
    logger = logging.getLogger(__name__)
    
    logger.info("Loading dataset...")
    # Read the CSV file
    df = pd.read_csv("/scratch/is2431/python_methods_comapred.csv")
    
    # Convert to Hugging Face Dataset if needed
    ds = Dataset.from_pandas(df)
    
    # Extract labels 
    train_labels = np.array(ds['target'])
    
    # Calculate class distribution
    unique_classes, class_counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    
    # Print class distribution statistics
    logger.info("\nClass Distribution Statistics:")
    for class_label, count in zip(unique_classes, class_counts):
        percentage = (count / total_samples) * 100
        logger.info(f"Class {class_label}: {count} samples ({percentage:.2f}%)")
    
    # Calculate balanced class weights
    balanced_weights = compute_balanced_class_weights(train_labels)
    
    logger.info("\nBalanced Class Weights:")
    for class_label, weight in balanced_weights.items():
        logger.info(f"Class {class_label}: {weight:.4f}")
    
    # Calculate alternative weights based on inverse class frequencies
    class_frequencies = class_counts / total_samples
    inverse_freq_weights = 1 / class_frequencies
    # Normalize weights to have mean = 1
    inverse_freq_weights = inverse_freq_weights / inverse_freq_weights.mean()
    
    logger.info("\nInverse Frequency Weights (normalized):")
    for class_label, weight in zip(unique_classes, inverse_freq_weights):
        logger.info(f"Class {class_label}: {weight:.4f}")
    
    # Calculate pos_weight for BCEWithLogitsLoss
    # Assuming binary classification with 0 and 1 classes
    neg_samples = class_counts[0]  # Assuming 0 is negative class
    pos_samples = class_counts[1]  # Assuming 1 is positive class
    pos_weight = neg_samples / pos_samples
    
    logger.info(f"\nRecommended pos_weight for BCEWithLogitsLoss: {pos_weight:.4f}")
    
    # Calculate effective sample weights for each class
    effective_weights = {
        'balanced': balanced_weights,
        'inverse_freq': dict(zip(unique_classes, inverse_freq_weights))
    }
    
    # Calculate median sample weights for each method
    logger.info("\nExample per-sample weights when using each method:")
    for method, weights in effective_weights.items():
        sample_weights = np.array([weights[label] for label in train_labels])
        logger.info(f"{method.title()} Weights - Median: {np.median(sample_weights):.4f}, "
                   f"Mean: {np.mean(sample_weights):.4f}, "
                   f"Std: {np.std(sample_weights):.4f}")
    
    return {
        'class_distribution': dict(zip(unique_classes, class_counts)),
        'balanced_weights': balanced_weights,
        'inverse_freq_weights': dict(zip(unique_classes, inverse_freq_weights)),
        'pos_weight': pos_weight,
        'class_ratios': {
            'neg_pos_ratio': neg_samples / pos_samples,
            'pos_neg_ratio': pos_samples / neg_samples
        }
    }

def main():
    logger = setup_logging()
    logger.info("Starting class weight calculation...")
    
    try:
        weights = load_and_analyze_dataset()
        
        logger.info("\nFinal Summary:")
        logger.info("==============")
        logger.info(f"Class Distribution: {weights['class_distribution']}")
        logger.info(f"Balanced Class Weights: {weights['balanced_weights']}")
        logger.info(f"Inverse Frequency Weights: {weights['inverse_freq_weights']}")
        logger.info(f"BCE pos_weight: {weights['pos_weight']}")
        logger.info("\nClass Ratios:")
        logger.info(f"Negative/Positive Ratio: {weights['class_ratios']['neg_pos_ratio']:.4f}")
        logger.info(f"Positive/Negative Ratio: {weights['class_ratios']['pos_neg_ratio']:.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
