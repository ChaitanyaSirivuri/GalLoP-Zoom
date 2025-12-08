# GalLoP-Zoom: Learning Global and Local Prompts for Vision-Language Models on MuSciClaims

This project implements the GalLoP (Global and Local Prompts) strategy for the MuSciClaims benchmark. It uses a two-stage approach to improve VLM performance:
1.  **Evidence Localization**: Using GroundingDINO to find relevant panels in figures.
2.  **GalLoP Prompt Training**: Fine-tuning VLMs (Qwen2.5-VL / Llava-Next) using claim-specific (Local) and caption-specific (Global) prompts.

## Setup & Installation
```bash
uv sync
```
Or use the provided `COLAB_SETUP.md` for running on Google Colab.

## Workflow

### 1. Data Preparation
First, download the images from the MuSciClaims dataset:
```bash
uv run download_figures.py
```
This downloads images to `paper_figures/`.

### 2. Evidence Localization (Stage 1)
**Note:** The localized panels (crops) are pre-calculated and located in `localized_features/`. We skip running the computationally expensive localization script in the default workflow to save time.


### 3. GalLoP Prompt Training (Stage 2)
Train the VLM to learn "GalLoP" prompts (Soft Prompts via PEFT). This keeps the model frozen and only learns to adapt to the claim/caption structure.

**For Qwen2.5-VL:**
```bash
uv run train_gallop.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --epochs 3
```

**For LLaVA-NeXT:**
```bash
uv run train_gallop.py --model_id llava-hf/llava-v1.6-mistral-7b-hf --epochs 3
```

### 4. Benchmarking
We follow a 4-step evaluation process. Run `benchmark/benchmark_gallop.py` with different flags.

**Step 1: Base Benchmark (Full Figure, No Training)**
Run the original model on the full figures.
```bash
uv run benchmark/benchmark_gallop.py --model_id Qwen/Qwen2.5-VL-7B-Instruct
```

**Step 2: Localized Only (Evidence Localization)**
Run the original model on the **localized crops** (from Stage 1). This tests if zooming in helps.
```bash
uv run benchmark/benchmark_gallop.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --use_localized
```

**Step 3: GalLoP Only (Trained Prompts)**
Run the **GalLoP-trained** model on full figures.
```bash
uv run benchmark/benchmark_gallop.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --use_gallop --adapter_path gallop_checkpoints
```

**Step 4: GalLoP + Localized Prompts (Full Pipeline)**
Run the **GalLoP-trained** model (which learned Global/Local prompts) on the **Localized Features (Crops)**. This combines both improvements:
- **Visual**: GroundingDINO crops the image (Localized Feature).
- **Textual**: GalLoP uses the Claim (Local Prompt) and Caption (Global Prompt).
```bash
uv run benchmark/benchmark_gallop.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --use_gallop --use_localized --adapter_path gallop_checkpoints
```

## Results Interpretation
- **Base vs Localized**: Shows the impact of GroundingDINO localization.
- **Base vs GalLoP**: Shows the impact of prompt training.
- **Localized + GalLoP**: Shows the combined synergistic effect.

**Visualizing Results**:
You can inspect the `localized_features/` folder to see how GroundingDINO cropped the panels compared to the original `paper_figures/`.

## Appendix: Reproducing Localization
If you need to regenerate the panel crops from scratch (Stage 1), you can run the localization script. **Warning: This uses GroundingDINO and is very slow.**
```bash
uv run localize_panels.py
```
