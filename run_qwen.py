
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
    output_dir = "localized_data"
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
    # 2. Load Manifest
    manifest_path = "dataset_manifest.json"
    if not os.path.exists(manifest_path):
        print("Manifest not found! Please run 'download_and_survey.py' first.")
        # For verification/testing if manifest strictly required:
        return

    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    # User limit for testing (can be removed later)
    LIMIT = 5 
    if LIMIT:
        print(f"Limiting to first {LIMIT} images for verification.")
        manifest = manifest[:LIMIT]
    
    # 3. Processing Loop
    metadata = []
    
    print("Starting Qwen Extraction...")
    for item in tqdm(manifest):
        fname = item["original_filename"]
        local_path = item["local_path"]
        target_panels = item["panels"]
        
        print(f"Processing {fname} -> Panels: {target_panels}")

        try:
            # Check file exists
            if not os.path.exists(local_path):
                print(f"  File missing: {local_path}")
                continue
            
            # Prepare Image
            # Qwen handles 'image' path in messages
            
            # We will ask for ALL panels in one prompt to save inference time?
            # Or one by one? One by one is safer for parsing specific labels.
            # Let's try iterating.
            
            image_source = Image.open(local_path).convert("RGB")
            w, h = image_source.size
            img_cv = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)

            PADDING = 20 # Safety margin

            for char in target_panels:
                # Prompt: Ask for inclusive box to avoid cutting labels
                prompt_text = f"Detect the inclusive bounding box of the subfigure Panel {char}, ensuring all axes, labels, and ticks are included. Return the result in JSON format with key 'bbox_2d'."
                
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
                
                # Parsing Logic
                save_params = None # (x1, y1, x2, y2)
                
                import re
                
                # 1. Try finding JSON block
                # Look for [ ... ] or { ... }
                json_match = re.search(r"(\[.*\]|\{.*\})", output_text, re.DOTALL)
                if json_match:
                    try:
                        import json
                        # Clean up markdown code blocks if present
                        json_str = json_match.group(1).replace("```json", "").replace("```", "")
                        data = json.loads(json_str)
                        
                        if isinstance(data, list): data = data[0]
                        
                        if "bbox_2d" in data:
                            bbox = data["bbox_2d"]
                            # Qwen2-VL JSON box seems to be [xmin, ymin, xmax, ymax] absolute vals based on logs
                            # e.g. [2143, 75, 3098, 496]
                            x1, y1, x2, y2 = bbox
                            save_params = (int(x1), int(y1), int(x2), int(y2))
                    except Exception as e_json:
                        print(f"  JSON Parse Error: {e_json}")

                # 2. Try the special token format if JSON failed
                if save_params is None and "<|box_start|>" in output_text:
                    try:
                        segment = output_text.split("<|box_start|>")[1].split("<|box_end|>")[0]
                        match = re.search(r"\((\d+),(\d+)\),\((\d+),(\d+)\)", segment)
                        if match:
                            y1_n, x1_n, y2_n, x2_n = map(int, match.groups())
                            x1 = int((x1_n / 1000) * w)
                            y1 = int((y1_n / 1000) * h)
                            x2 = int((x2_n / 1000) * w)
                            y2 = int((y2_n / 1000) * h)
                            save_params = (x1, y1, x2, y2)
                    except:
                        pass

                # If we got coordinates, Save
                if save_params:
                    x1, y1, x2, y2 = save_params
                    
                    # Apply Padding
                    x1 = max(0, x1 - PADDING)
                    y1 = max(0, y1 - PADDING)
                    x2 = min(w, x2 + PADDING)
                    y2 = min(h, y2 + PADDING)
                    
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
                    print(f"  Saved {save_name} (Box: {save_params}, Padded)")
                    
                    metadata.append({
                        "original_image": fname,
                        "panel": char,
                        "crop_path": save_name,
                        "bbox": [x1, y1, x2, y2]
                    })
                else:
                    print(f"  No box found for {char}")

        except Exception as e:
            print(f"Failed {fname}: {e}")

    with open("panel_crops_qwen_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    process_all_images_qwen()
