# vulnLLM 🛡️  
**An LLM-based System for Detecting Code Vulnerabilities**

vulnLLM is a fine-tuned large language model designed to detect security vulnerabilities in Python source code. Built on top of InternLM2.5 and Phi3.5, this system combines deep learning with software security practices to enable early detection of unsafe code patterns.

---

## 🚀 Features

- **LLM-based Vulnerability Detection**: Built on InternLM2.5 (1.8B and 7B Chat variants), fine-tuned with real-world vulnerable and secure code samples.
- **Custom Classifier Head**: Added on top of the base model to output binary vulnerability labels.
- **Memory-Efficient Training**: Uses gradient checkpointing, accumulation, and NVML tracking to train efficiently on limited GPU resources.
- **Dataset Integration**: Uses a cleaned and normalized version of the [Devign](https://huggingface.co/datasets/DetectVul/devign) dataset.
- **Evaluation Tools**: Includes validation scripts and metrics (F1, precision, recall) for benchmarking performance.

---

## 🧠 Model Architecture

- Base: InternLM2.5 1.8B/7B Chat
- Fine-tuning: Supervised classification with `func_clean` → `vulnerability` (0 or 1)
- Add-on: Classifier head (`MLP` or `linear`) on pooled representation

---

## 📊 Dataset

| Split        | # Samples |
|--------------|-----------|
| Train        | 21,900    |
| Validation   | 2,700     |

Each row includes:
- `func`: Raw function code
- `func_clean`: Normalized version
- `target`: 1 if vulnerable, 0 otherwise
- `project`, `commit_id`, `vul_lines`, etc. for metadata
