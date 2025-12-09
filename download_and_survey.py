
import os
import json
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def main():
    print("Loading dataset...")
    try:
        ds = load_dataset("StonyBrookNLP/MuSciClaims", split="test")
    except:
        ds = load_dataset("StonyBrookNLP/MuSciClaims", split="train")
        
    print(f"Dataset Loaded. Total claim pairs: {len(ds)}")
    
    unique_images = {} # fname -> set(panels)
    
    print("Aggregating unique images and panels...")
    for item in tqdm(ds):
        fname = item.get('associated_figure_filepath')
        if not fname: continue
        
        if fname not in unique_images:
            unique_images[fname] = set()
            
        # Parse panels
        raw_panels = item.get('associated_figure_panels', [])
        if isinstance(raw_panels, list):
            for p in raw_panels:
                # Clean "Panel A" -> "A"
                clean_p = p.replace("Panel ", "").strip()
                if len(clean_p) == 1 and clean_p.isalpha():
                    unique_images[fname].add(clean_p)
                else:
                    # Keep as is if weird? Or ignore? 
                    # Let's keep strict for now to avoid noise
                    pass

    print(f"Found {len(unique_images)} unique images.")
    
    # Download Phase
    output_dir = "raw_images"
    os.makedirs(output_dir, exist_ok=True)
    
    manifest = []
    
    print(f"Downloading images to {output_dir}...")
    
    success_count = 0
    fail_count = 0
    
    for fname, panels_set in tqdm(unique_images.items()):
        try:
            # Download from Hub
            # We use hf_hub_download. 
            # To verify we have it "locally", we can copy it or look at the cache path.
            # Let's copy to be explicit as requested.
            cached_path = hf_hub_download(
                repo_id="StonyBrookNLP/MuSciClaims",
                filename=fname,
                repo_type="dataset"
            )
            
            # Destination path
            clean_name = fname.replace("/", "_")
            dest_path = os.path.join(output_dir, clean_name)
            
            if not os.path.exists(dest_path):
                shutil.copy(cached_path, dest_path)
            
            sorted_panels = sorted(list(panels_set))
            if not sorted_panels:
                sorted_panels = list("ABCDEFGHIJ") # Fallback
            
            manifest.append({
                "original_filename": fname,
                "local_path": dest_path,
                "panels": sorted_panels
            })
            success_count += 1
            
        except Exception as e:
            print(f"Failed to download {fname}: {e}")
            fail_count += 1

    # Save Manifest
    with open("dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n--- Summary ---")
    print(f"Total Unique Images: {len(unique_images)}")
    print(f"Successfully Downloaded: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Manifest saved to 'dataset_manifest.json'")
    
    # Print Distribution
    print("Sample entries:")
    for m in manifest[:5]:
        print(f"  {m['original_filename']} -> {len(m['panels'])} panels: {m['panels']}")

if __name__ == "__main__":
    main()
