# MuSciClaims Benchmark

Benchmark suite for evaluating Vision-Language Models on the MuSciClaims scientific claim verification dataset.

## Setup

### Prerequisites

1. Install required packages:
```bash
pip install transformers datasets torch pillow tqdm
pip install qwen-vl-utils  # For Qwen models
```

2. Ensure you have the localized features extracted:
```bash
python localize_panels.py
```

## Running Benchmarks

### Individual Model Benchmarks

**Qwen2.5-VL-7B-Instruct:**
```bash
# Windows PowerShell
uv run python -m benchmark.benchmark_qwen25_vl

# Linux/Mac
python -m benchmark.benchmark_qwen25_vl
```

**Qwen3-VL-30B-A3B-Instruct:**
```bash
# Windows PowerShell
uv run python -m benchmark.benchmark_qwen3_vl

# Linux/Mac
python -m benchmark.benchmark_qwen3_vl
```

**LLaVA-Next (Mistral-7B):**
```bash
# Windows PowerShell
uv run python -m benchmark.benchmark_llava_next

# Linux/Mac
python -m benchmark.benchmark_llava_next
```

### Run All Benchmarks
```bash
# Windows PowerShell
uv run python -m benchmark.run_all

# Linux/Mac
python -m benchmark.run_all
```

## Output

Results are saved to the `results/` directory:

- `{model}_metrics_{timestamp}.csv` - Per-class and overall metrics
- `{model}_predictions_{timestamp}.csv` - Detailed predictions for each sample
- `{model}_results_{timestamp}.json` - Full results with metadata
- `comparison_report_{timestamp}.csv` - Comparison across all models

## Metrics

The benchmark computes:
- **Precision, Recall, F1** for each class (SUPPORT, NEUTRAL, CONTRADICT)
- **Overall (Macro-averaged)** Precision, Recall, F1
- **Accuracy**

## File Structure

```
benchmark/
├── __init__.py           # Package initialization
├── data_loader.py        # Load images from localized_features
├── prompts.py            # Prompt templates for D (Direct Decision)
├── metrics.py            # Metrics calculation and saving
├── benchmark_qwen25_vl.py  # Qwen2.5-VL benchmark
├── benchmark_qwen3_vl.py   # Qwen3-VL benchmark
├── benchmark_llava_next.py # LLaVA-Next benchmark
└── run_all.py            # Run all benchmarks
```

## GPU Requirements

- **Qwen2.5-VL-7B**: ~16GB VRAM
- **Qwen3-VL-30B-A3B**: ~40GB VRAM (MoE architecture)
- **LLaVA-Next-7B**: ~16GB VRAM

For lower VRAM, you can modify the scripts to use 4-bit quantization.
