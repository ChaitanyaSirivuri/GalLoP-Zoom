"""
Data loader for MuSciClaims benchmark.
Loads images from localized_features based on associated panels.
"""

import os
from PIL import Image
from datasets import load_dataset
from typing import List, Tuple, Dict, Optional


# Directory containing localized panel images
LOCALIZED_FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "localized_features")
PAPER_FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_figures")


def extract_panel_id(panel_str: str) -> str:
    """
    Extract the panel ID from panel string.
    Examples: 
        "Panel I" -> "I"
        "panel A" -> "A"
        "(b)" -> "B"
        "Figure 2C" -> "Figure 2C"
        "ALL Panels" -> "All Panels"
        "No panel markings" -> "No panel markings"
    """
    panel_str = str(panel_str).strip()
    
    # Handle special cases - return as-is with spaces for filename matching
    special_cases_map = {
        "all panels": "All Panels",
        "no panel markings": "No panel markings",
        "full figure": "Full Figure",
    }
    if panel_str.lower() in special_cases_map:
        return special_cases_map[panel_str.lower()]
    
    # Handle "Figure 2C" format -> keep as "Figure 2C"
    if panel_str.lower().startswith("figure "):
        return panel_str  # Keep original format
    
    # Remove common prefixes for regular panels
    for prefix in ["Panel ", "panel ", "Subfigure ", "subfigure "]:
        if panel_str.startswith(prefix):
            panel_str = panel_str[len(prefix):]
            break
    
    # Remove parentheses
    panel_str = panel_str.strip("()")
    
    # Return uppercase letter/identifier
    return panel_str.strip().upper()


def get_image_path_for_panel(figure_filepath: str, panel: str) -> Optional[str]:
    """
    Get the localized image path for a specific panel.
    
    Image naming conventions in localized_features:
    - For "Panel I": bio_..._images_figure_1_I.jpg
    - For "All Panels": bio_..._images_figure_1_All Panels.jpg (or .png)
    - For "Figure 2C": bio_..._images_figure_2_Figure 2C.jpg
    - For "No panel markings": bio_..._images_figure_2_No panel markings.jpg
    """
    # Get base filename without extension
    filename = os.path.basename(figure_filepath)
    
    # Handle jacs_data -> chem prefix fix
    if filename.startswith("jacs_data_10.1021_"):
        filename = filename.replace("jacs_data_10.1021_", "chem_10.1021_")
    
    base_name = os.path.splitext(filename)[0]
    
    # Extract panel ID
    panel_id = extract_panel_id(panel)
    
    # Try different filename patterns and extensions
    possible_names = []
    
    # Pattern 1: base_PANELID (e.g., bio_..._I.jpg)
    possible_names.append(f"{base_name}_{panel_id}")
    
    # Pattern 2: For special cases with spaces
    if panel_id in ["All Panels", "No panel markings", "Full Figure"]:
        possible_names.append(f"{base_name}_{panel_id}")
    
    # Pattern 3: For "Figure XY" format
    if panel.lower().startswith("figure "):
        possible_names.append(f"{base_name}_{panel}")
    
    # Try each pattern with different extensions
    extensions = [".jpg", ".png", ".jpeg"]
    
    for name in possible_names:
        for ext in extensions:
            path = os.path.join(LOCALIZED_FEATURES_DIR, f"{name}{ext}")
            if os.path.exists(path):
                return path
    
    return None


def get_fallback_image_path(figure_filepath: str) -> Optional[str]:
    """
    Get the original paper figure if localized panel not found.
    """
    filename = os.path.basename(figure_filepath)
    
    # Handle jacs_data -> chem prefix fix
    if filename.startswith("jacs_data_10.1021_"):
        filename = filename.replace("jacs_data_10.1021_", "chem_10.1021_")
    
    path = os.path.join(PAPER_FIGURES_DIR, filename)
    if os.path.exists(path):
        return path
    
    return None


def load_image_for_claim(figure_filepath: str, panels: List[str]) -> Tuple[Optional[Image.Image], str]:
    """
    Load the appropriate image for a claim based on associated panels.
    
    Returns:
        Tuple of (PIL Image or None, image_path or error message)
    """
    # If no panels specified, use the full figure
    if not panels:
        fallback = get_fallback_image_path(figure_filepath)
        if fallback:
            return Image.open(fallback).convert("RGB"), fallback
        return None, "No image found (no panels specified)"
    
    # Try to find localized panel image for the first panel
    # (Most claims reference a single panel)
    for panel in panels:
        image_path = get_image_path_for_panel(figure_filepath, panel)
        if image_path:
            return Image.open(image_path).convert("RGB"), image_path
    
    # Fallback to original paper figure
    fallback = get_fallback_image_path(figure_filepath)
    if fallback:
        return Image.open(fallback).convert("RGB"), fallback
    
    return None, f"No image found for panels: {panels}"


def load_dataset_with_images() -> List[Dict]:
    """
    Load the MuSciClaims dataset and prepare it with image paths.
    
    Returns:
        List of dictionaries with claim data and image information.
    """
    print("Loading MuSciClaims dataset...")
    ds = load_dataset("StonyBrookNLP/MuSciClaims")
    df = ds['test']
    
    print(f"Dataset loaded with {len(df)} records")
    
    data = []
    missing_images = 0
    
    for idx in range(len(df)):
        row = df[idx]
        
        figure_filepath = row['associated_figure_filepath']
        panels = row.get('associated_figure_panels', [])
        
        # Get image path
        image, image_path = load_image_for_claim(figure_filepath, panels)
        
        if image is None:
            missing_images += 1
            image_path = None
        else:
            image.close()  # Close to save memory, will reload when needed
        
        data.append({
            "index": idx,
            "claim_text": row['claim_text'],
            "caption": row.get('caption', ''),
            "label_3class": row['label_3class'],
            "label_2class": row.get('label_2class', ''),
            "associated_figure_filepath": figure_filepath,
            "associated_figure_panels": panels,
            "image_path": image_path,
            "paper_id": row.get('paper_id', ''),
            "domain": row.get('domain', ''),
        })
    
    print(f"Prepared {len(data)} samples, {missing_images} missing images")
    return data


def get_image_for_sample(sample: Dict) -> Optional[Image.Image]:
    """
    Load the image for a sample.
    
    Args:
        sample: Dictionary from load_dataset_with_images()
        
    Returns:
        PIL Image or None
    """
    if sample['image_path'] is None:
        return None
    
    try:
        return Image.open(sample['image_path']).convert("RGB")
    except Exception as e:
        print(f"Error loading image {sample['image_path']}: {e}")
        return None


if __name__ == "__main__":
    # Test the data loader
    data = load_dataset_with_images()
    
    # Show some examples
    print("\n--- Sample Data ---")
    for i in range(min(5, len(data))):
        sample = data[i]
        print(f"\n[{i}] Claim: {sample['claim_text'][:100]}...")
        print(f"    Label: {sample['label_3class']}")
        print(f"    Panels: {sample['associated_figure_panels']}")
        print(f"    Image: {sample['image_path']}")
