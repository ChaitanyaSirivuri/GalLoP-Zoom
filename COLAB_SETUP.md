# Running MuSciClaims Benchmark on Google Colab

## Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Go to `Runtime` → `Change runtime type` → Select `T4 GPU` or `A100 GPU` (recommended for faster inference)

## Step 2: Clone the Repository
```python
!git clone https://github.com/ChaitanyaSirivuri/GalLoP-Zoom.git
%cd GalLoP-Zoom
```

## Step 3: Install Dependencies
```python
!pip install -q transformers datasets torch pillow tqdm qwen-vl-utils accelerate
```

## Step 4: Download Dataset Images (Optional - Run Only Once)
```python
# This downloads all paper figures to paper_figures/ folder
!python download_figures.py
```

## Step 5: Localize Figure Panels (Optional - Run Only Once)
```python
# Install GroundingDINO dependencies
!pip install -q supervision groundingdino-py

# Run panel localization
!python localize_panels.py
```

## Step 6: Run Benchmarks

### Option A: Run Individual Model

**Qwen2.5-VL-7B:**
```python
!python -m benchmark.benchmark_qwen25_vl
```

**Qwen3-VL-8B:**
```python
!python -m benchmark.benchmark_qwen3_vl
```

**LLaVA-Next:**
```python
!python -m benchmark.benchmark_llava_next
```

### Option B: Run All Models
```python
!python -m benchmark.run_all
```

## Step 7: View Results

### List result files:
```python
!ls -lh results/
```

### View metrics CSV:
```python
import pandas as pd

# View Qwen2.5-VL metrics
df = pd.read_csv("results/qwen25_vl_metrics_*.csv")  # Replace * with actual timestamp
print(df)
```

### Download results to your computer:
```python
from google.colab import files

# Download specific file
files.download('results/qwen25_vl_metrics_<timestamp>.csv')

# Or zip all results
!zip -r results.zip results/
files.download('results.zip')
```

## Step 8: Monitor GPU Usage (Optional)
```python
# Check GPU memory
!nvidia-smi

# Monitor during inference
import subprocess
import time

def monitor_gpu():
    for _ in range(10):
        subprocess.run(['nvidia-smi'])
        time.sleep(5)

# Run in background
import threading
thread = threading.Thread(target=monitor_gpu)
thread.start()
```

## Quick Start: Complete Notebook Cell

Copy this into a single Colab cell to run everything:

```python
# Setup
!git clone https://github.com/ChaitanyaSirivuri/GalLoP-Zoom.git
%cd GalLoP-Zoom
!pip install -q transformers datasets torch pillow tqdm qwen-vl-utils accelerate

# Download and localize (if needed)
!python download_figures.py
!pip install -q supervision groundingdino-py
!python localize_panels.py

# Run benchmark (choose one)
!python -m benchmark.benchmark_qwen25_vl  # Fastest, ~16GB VRAM
# !python -m benchmark.benchmark_qwen3_vl  # Medium, ~16GB VRAM
# !python -m benchmark.benchmark_llava_next  # Alternative, ~16GB VRAM
# !python -m benchmark.run_all  # Run all three models

# View results
import pandas as pd
!ls results/
```

## Troubleshooting

### Out of Memory (OOM) Error
- Use T4 GPU for 7B/8B models
- Use A100 GPU (Colab Pro) for better performance
- Run models one at a time instead of `run_all.py`

### Dataset Download Issues
- Dataset automatically downloads from HuggingFace
- If it fails, check internet connection and retry

### Model Download Slow
- First run downloads models (~15-30GB total)
- Subsequent runs use cached models
- Be patient, can take 10-20 minutes on first run

### Missing Localized Features
- Run `localize_panels.py` first
- This creates cropped panel images in `localized_features/`
- Falls back to full paper figures if panels not found

## Expected Runtime

| Task | T4 GPU | A100 GPU |
|------|--------|----------|
| Install dependencies | ~2 min | ~2 min |
| Download images | ~5 min | ~5 min |
| Localize panels | ~30 min | ~15 min |
| Benchmark (per model) | ~2-3 hours | ~30-45 min |
| All three models | ~6-9 hours | ~1.5-2 hours |

## Tips for Colab
- Enable background execution: `Tools` → `Settings` → `Miscellaneous` → Check "Automatically run all enabled code blocks"
- Keep tab active to prevent disconnection
- Use Colab Pro for longer runtime and better GPUs
- Save intermediate results to Google Drive to avoid losing progress
