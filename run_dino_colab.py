
import os
import torch
import cv2
import numpy as np
import json
from PIL import Image
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm
import sys

# Ensure GroundingDINO is in path (common issue in Colab/Repos)
if os.path.isdir("GroundingDINO"):
    sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

try:
    from groundingdino.util.inference import load_model, load_image, predict
except ImportError:
    print("ERROR: Could not import GroundingDINO. Make sure you ran 'setup_colab.sh'!")
    sys.exit(1)

def process_all_images():
    output_dir = "processed_crops"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model
    config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print("Error: Model files not found. Please run setup_colab.sh first.")
        return

    print("Loading GroundingDINO...")
    model = load_model(config_path, weights_path)

    # 2. Get Unique Images from Dataset
    print("Loading dataset metadata...")
    # NOTE: MuSciClaims on HF only lists 'test' split in default config. 
    # We use 'test' for now to ensure this runs. 
    try:
        ds = load_dataset("StonyBrookNLP/MuSciClaims", split="test")
    except Exception as e:
        print(f"Error loading 'test' split: {e}. Trying 'train'...")
        ds = load_dataset("StonyBrookNLP/MuSciClaims", split="train") 
    
    unique_images = {} 
    
    print("Identifying unique images...")
    for item in tqdm(ds):
        fname = item['associated_figure_filepath']
        if fname not in unique_images:
            unique_images[fname] = item 
            
    print(f"Found {len(unique_images)} unique images for processing.")
    
    # 3. Process Batch
    # Prompt: We look for all alphabetical panels reasonably expected
    chars = "ABCDEFGHIJKLM" 
    ALL_PANELS_PROMPT = " . ".join([f"Panel {char}" for char in chars])
    
    # Thresholds
    BOX_ERR = 0.25 # Slightly looser to catch everything
    TEXT_ERR = 0.25

    metadata = []

    print("Starting processing...")
    for fname, item in tqdm(unique_images.items()):
        try:
            # Download
            local_path = hf_hub_download(
                repo_id="StonyBrookNLP/MuSciClaims",
                filename=fname,
                repo_type="dataset"
            )
            
            image_source, image_tensor = load_image(local_path)
            
            # Predict
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=ALL_PANELS_PROMPT,
                box_threshold=BOX_ERR,
                text_threshold=TEXT_ERR
            )
            
            # Save Crops
            h, w, _ = image_source.shape
            
            for box, phrase in zip(boxes, phrases):
                # Clean phrase to get ID (e.g. "Panel A" -> "A")
                panel_id = phrase.replace("Panel", "").strip().upper() 
                # Basic validation: ensure it's a single letter or close to it
                if len(panel_id) > 2: continue 
                
                # Unnormalize box
                cx, cy, bw, bh = box * torch.Tensor([w, h, w, h])
                x1 = int(cx - bw/2)
                y1 = int(cy - bh/2)
                x2 = int(cx + bw/2)
                y2 = int(cy + bh/2)
                
                # Clip
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Filter tiny boxes (noise)
                if (x2 - x1) < 10 or (y2 - y1) < 10: continue

                crop = image_source[y1:y2, x1:x2]
                
                # Naming
                clean_name = fname.replace("/", "_").replace(".jpg", "").replace(".png", "")
                save_name = f"{clean_name}_Panel_{panel_id}.jpg"
                save_path = os.path.join(output_dir, save_name)
                
                cv2.imwrite(save_path, crop)
                
                metadata.append({
                    "original_image": fname,
                    "panel": panel_id,
                    "crop_path": save_name,
                    "bbox": [x1, y1, x2, y2]
                })
                
        except Exception as e:
            print(f"Failed {fname}: {e}")

    with open("panel_crops_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Done! {len(metadata)} panels extracted. Metadata saved to panel_crops_metadata.json")

if __name__ == "__main__":
    process_all_images()
