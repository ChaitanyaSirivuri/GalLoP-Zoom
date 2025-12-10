"""
Script to localize figure panels from MuSciClaims dataset images using GroundingDINO.
This extracts and crops the specific panels referenced in claims for better benchmarking.
"""

import os
import sys
import json
import cv2
import torch
from PIL import Image
from datasets import load_dataset

# Add GroundingDINO to path (adjust path as needed)
GROUNDING_DINO_PATH = os.path.join(os.path.dirname(__file__), "GroundingDINO")
sys.path.insert(0, GROUNDING_DINO_PATH)

from groundingdino.util.inference import load_model, load_image, predict, annotate

# Configuration
CONFIG_PATH = os.path.join(GROUNDING_DINO_PATH, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(GROUNDING_DINO_PATH, "weights/groundingdino_swint_ogc.pth")
INPUT_DIR = "paper_figures"
OUTPUT_DIR = "localized_features"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_panel_prompt(panels):
    """
    Convert panel labels to a text prompt for GroundingDINO.
    Panels are typically labeled as 'A', 'B', 'C', etc. or '(a)', '(b)', etc.
    """
    if not panels:
        return None
    
    # Create prompt for each panel
    # GroundingDINO expects phrases separated by " . "
    panel_prompts = []
    for panel in panels:
        # Handle different panel formats
        panel_str = str(panel).strip().upper()
        # Add various formats the model might recognize
        panel_prompts.append(f"panel {panel_str}")
        panel_prompts.append(f"subfigure {panel_str}")
        panel_prompts.append(f"({panel_str.lower()})")
        panel_prompts.append(f"{panel_str}")
    
    return " . ".join(panel_prompts)


def extract_panel_id(panel_str):
    """
    Extract the panel ID from panel string.
    Examples: 
        "Panel I" -> "I"
        "panel A" -> "A"
        "(b)" -> "B"
        "Figure 2C" -> "2C"
        "ALL Panels" -> "ALL_PANELS"
        "No panel markings" -> "NO_PANEL_MARKINGS"
    """
    panel_str = str(panel_str).strip()
    
    # Handle special cases - return as-is with underscores
    special_cases = ["ALL Panels", "No panel markings", "All Panels", "all panels", 
                     "No Panel Markings", "no panel markings", "Full Figure", "full figure"]
    for special in special_cases:
        if panel_str.lower() == special.lower():
            return panel_str.upper().replace(" ", "_")
    
    # Handle "Figure 2C" format -> extract "2C"
    if panel_str.lower().startswith("figure "):
        panel_str = panel_str[7:]  # Remove "Figure "
        return panel_str.strip().upper()
    
    # Remove common prefixes
    for prefix in ["Panel ", "panel ", "Subfigure ", "subfigure "]:
        if panel_str.startswith(prefix):
            panel_str = panel_str[len(prefix):]
            break
    
    # Remove parentheses
    panel_str = panel_str.strip("()")
    
    # Return uppercase letter/identifier, replace spaces with underscores
    result = panel_str.strip().upper().replace(" ", "_")
    return result if result else "FULL"


def crop_and_save_panel(image_path, boxes, logits, phrases, panels, output_path):
    """
    Crop the detected panel regions and save them.
    Returns list of saved crop paths.
    """
    saved_crops = []
    
    # Load original image for cropping
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return saved_crops
    
    h, w = image.shape[:2]
    
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    # Map detected phrases to panel IDs from the requested panels
    panel_ids = [extract_panel_id(p) for p in panels] if panels else []
    
    for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
        # Try to match the detected phrase to a panel ID
        detected_panel_id = extract_panel_id(phrase)
        
        # Only save if it matches one of the requested panels (or if no specific panels requested)
        if panels and detected_panel_id not in panel_ids:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        # box format: [cx, cy, w, h] normalized
        cx, cy, bw, bh = box
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Crop the region
        crop = image[y1:y2, x1:x2]
        
        if crop.size > 0:
            # Save cropped panel with panel ID in filename
            # Format: originalname_PANELID.jpg (e.g., bio_PIIS0092867425000455_images_figure_1_I.jpg)
            crop_path = os.path.join(
                OUTPUT_DIR, 
                f"{base_name}_{detected_panel_id}.jpg"
            )
            
            # Skip if already exists
            if os.path.exists(crop_path):
                print(f"    Skipping (already exists): {os.path.basename(crop_path)}")
                saved_crops.append({
                    "path": crop_path,
                    "phrase": phrase,
                    "panel_id": detected_panel_id,
                    "confidence": float(logit),
                    "box": [x1, y1, x2, y2],
                    "box_normalized": [float(cx), float(cy), float(bw), float(bh)],
                    "skipped": True
                })
                continue
            
            cv2.imwrite(crop_path, crop)
            saved_crops.append({
                "path": crop_path,
                "phrase": phrase,
                "panel_id": detected_panel_id,
                "confidence": float(logit),
                "box": [x1, y1, x2, y2],
                "box_normalized": [float(cx), float(cy), float(bw), float(bh)],
                "skipped": False
            })
    
    return saved_crops


def process_dataset():
    """
    Process the MuSciClaims dataset and localize panels for each claim.
    """
    print("Loading GroundingDINO model...")
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow. Consider setting up CUDA.")
    
    # Load model
    model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=device)
    
    print("Loading MuSciClaims dataset...")
    ds = load_dataset("StonyBrookNLP/MuSciClaims")
    df = ds['test']
    
    print(f"Dataset loaded with {len(df)} records")
    
    # Track results
    results = []
    processed_images = set()  # Track which images we've already processed
    
    for row_index in range(len(df)):
        row = df[row_index]
        
        figure_path = row['associated_figure_filepath']
        panels = row.get('associated_figure_panels', [])
        claim_text = row.get('claim_text', '')
        
        # Get the local image path
        # Handle the path prefix fix (jacs_data -> chem)
        filename = os.path.basename(figure_path)
        if filename.startswith("jacs_data_10.1021_"):
            filename = filename.replace("jacs_data_10.1021_", "chem_10.1021_")
        
        local_image_path = os.path.join(INPUT_DIR, filename)
        
        if not os.path.exists(local_image_path):
            print(f"[{row_index}] Image not found: {local_image_path}")
            continue
        
        # Create unique key for this image + panels combination
        panels_key = "_".join(sorted([str(p) for p in panels])) if panels else "full"
        image_key = f"{filename}_{panels_key}"
        
        if image_key in processed_images:
            # Already processed this exact image+panels combination
            continue
        
        processed_images.add(image_key)
        
        # Generate prompt from panels
        if panels:
            text_prompt = get_panel_prompt(panels)
        else:
            # If no specific panels, try to detect all panels
            text_prompt = "panel A . panel B . panel C . panel D . panel E . panel F . subfigure . graph . chart . diagram"
        
        print(f"\n[{row_index}] Processing: {filename}")
        print(f"  Panels: {panels}")
        print(f"  Prompt: {text_prompt[:100]}...")
        
        try:
            # Load image for GroundingDINO
            image_source, image_tensor = load_image(local_image_path)
            
            # Run prediction
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=device
            )
            
            print(f"  Detected {len(boxes)} regions")
            
            if len(boxes) > 0:
                # Save annotated image for visualization
                annotated_frame = annotate(
                    image_source=image_source, 
                    boxes=boxes, 
                    logits=logits, 
                    phrases=phrases
                )
                annotated_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
                cv2.imwrite(annotated_path, annotated_frame)
                
                # Crop and save individual panels
                crops = crop_and_save_panel(
                    local_image_path, 
                    boxes.numpy() if hasattr(boxes, 'numpy') else boxes,
                    logits.numpy() if hasattr(logits, 'numpy') else logits,
                    phrases,
                    panels,
                    local_image_path
                )
                
                results.append({
                    "row_index": row_index,
                    "image_path": local_image_path,
                    "panels_requested": panels,
                    "claim_text": claim_text[:200] + "..." if len(claim_text) > 200 else claim_text,
                    "detections": crops,
                    "annotated_image": annotated_path
                })
                
                for crop in crops:
                    status = "skipped" if crop.get('skipped') else "saved"
                    print(f"    - {crop['panel_id']}: conf={crop['confidence']:.2f}, {status} -> {os.path.basename(crop['path'])}")
            else:
                print("  No panels detected")
                results.append({
                    "row_index": row_index,
                    "image_path": local_image_path,
                    "panels_requested": panels,
                    "claim_text": claim_text[:200] + "..." if len(claim_text) > 200 else claim_text,
                    "detections": [],
                    "error": "No panels detected"
                })
                
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "row_index": row_index,
                "image_path": local_image_path,
                "panels_requested": panels,
                "error": str(e)
            })
    
    # Save results summary
    results_path = os.path.join(OUTPUT_DIR, "localization_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n--- Processing Complete ---")
    print(f"Total images processed: {len(processed_images)}")
    print(f"Results saved to: {results_path}")
    print(f"Localized features saved to: {OUTPUT_DIR}/")
    
    return results


if __name__ == "__main__":
    # Check if GroundingDINO is installed
    if not os.path.exists(GROUNDING_DINO_PATH):
        print(f"Error: GroundingDINO not found at {GROUNDING_DINO_PATH}")
        print("\nPlease install GroundingDINO first:")
        print("  git clone https://github.com/IDEA-Research/GroundingDINO.git")
        print("  cd GroundingDINO")
        print("  pip install -e .")
        print("  mkdir weights && cd weights")
        print("  wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        sys.exit(1)
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Model weights not found at {WEIGHTS_PATH}")
        print("\nPlease download the weights:")
        print("  cd GroundingDINO/weights")
        print("  wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        sys.exit(1)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        print("Please run download_figures.py first to download the images.")
        sys.exit(1)
    
    process_dataset()
