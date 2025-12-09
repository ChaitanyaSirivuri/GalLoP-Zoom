
import os
import torch
import cv2
import json
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def process_all_images_qwen():
    output_dir = "processed_crops_qwen"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"Loading {model_id}...")
    
    # Use bfloat16 if A100/L4, else float16 or float32. T4 supports float16.
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Get Data
    print("Loading dataset metadata...")
    try:
        ds = load_dataset("StonyBrookNLP/MuSciClaims", split="test")
    except:
        ds = load_dataset("StonyBrookNLP/MuSciClaims", split="train")

    unique_images = {}
    LIMIT = 5 # User requested limit for testing

    print("Identifying unique images...")
    for item in tqdm(ds):
        fname = item['associated_figure_filepath']
        if fname not in unique_images:
            unique_images[fname] = item
            if len(unique_images) >= LIMIT:
                print(f"Reached limit of {LIMIT} unique images.")
                break
    
    # 3. Processing Loop
    chars = "ABCDEFGHIJ" # Look for these panels
    metadata = []
    
    print("Starting Qwen Extraction...")
    for fname, item in tqdm(unique_images.items()):
        try:
            # Download
            local_path = hf_hub_download(
                repo_id="StonyBrookNLP/MuSciClaims",
                filename=fname,
                repo_type="dataset"
            )
            
            # Prepare Image
            # Qwen handles 'image' path in messages
            
            # We will ask for ALL panels in one prompt to save inference time?
            # Or one by one? One by one is safer for parsing specific labels.
            # Let's try iterating.
            
            image_source = Image.open(local_path).convert("RGB")
            w, h = image_source.size
            img_cv = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)

            for char in chars:
                # Prompt tuning: Be explicit about "subfigure" to avoid just finding the text label.
                prompt_text = f"Detect the bounding box of the subfigure Panel {char}."
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": local_path},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                
                # Decode ONLY new tokens
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=False
                )[0]
                
                print(f"DEBUG: Raw Output for {char}: {output_text}")
                
                # Robust Regex Parsing for <|box_start|>(y1,x1),(y2,x2)<|box_end|>
                import re
                # Pattern: (number,number),(number,number)
                match = re.search(r"\((\d+),(\d+)\),\((\d+),(\d+)\)", output_text)
                
                if match:
                    try:
                        # Qwen2-VL Output is usually (y1, x1), (y2, x2) in [0, 1000] scale
                        # But let's verify if it's y,x or x,y.
                        # Standard Qwen2-VL is (y_min, x_min), (y_max, x_max)
                        y1_n, x1_n, y2_n, x2_n = map(int, match.groups())
                        
                        # Scale to image
                        x1 = int((x1_n / 1000) * w)
                        y1 = int((y1_n / 1000) * h)
                        x2 = int((x2_n / 1000) * w)
                        y2 = int((y2_n / 1000) * h)
                        
                        # Validate
                        if (x2 - x1) < 10 or (y2 - y1) < 10: 
                            print(f"  Skipping {char} (Too small)")
                            continue
                        
                        # Crop
                        crop = img_cv[y1:y2, x1:x2]
                        
                        clean_name = fname.replace("/", "_").replace(".jpg", "").replace(".png", "")
                        save_name = f"{clean_name}_Panel_{char}.jpg"
                        save_path = os.path.join(output_dir, save_name)
                        
                        cv2.imwrite(save_path, crop)
                        print(f"  Saved {save_name}")
                        
                        metadata.append({
                            "original_image": fname,
                            "panel": char,
                            "crop_path": save_name,
                            "bbox": [x1, y1, x2, y2]
                        })

                    except Exception as e_parse:
                        print(f"  Parse Error for {char}: {e_parse}")
                else:
                    # Fallback or just ignore
                    pass

        except Exception as e:
            print(f"Failed {fname}: {e}")

    with open("panel_crops_qwen_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    process_all_images_qwen()
