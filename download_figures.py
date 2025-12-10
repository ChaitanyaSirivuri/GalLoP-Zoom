"""
Script to download images from the MuSciClaims dataset to paper_figures folder.
"""

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import shutil

# Create the output directory
output_dir = "paper_figures"
os.makedirs(output_dir, exist_ok=True)

# Download the dataset
print("Loading dataset...")
ds = load_dataset("StonyBrookNLP/MuSciClaims")
df = ds['test']

print(f"Dataset loaded with {len(df)} records")

# Track unique figures to avoid duplicates
downloaded_figures = set()
success_count = 0
skip_count = 0
error_count = 0

# Download all associated figures
for row_index in range(len(df)):
    row = df[row_index]
    figure_path = row['associated_figure_filepath']
    
    # Skip if we've already downloaded this figure
    if figure_path in downloaded_figures:
        skip_count += 1
        continue
    
    downloaded_figures.add(figure_path)
    
    try:
        # Fix the path prefix: replace jacs_data_10.1021_ with chem_10.1021_ for actual files
        corrected_path = figure_path.replace("jacs_data_10.1021_", "chem_10.1021_")
        
        # Download the associated figure using corrected path
        local_path = hf_hub_download(
            repo_id="StonyBrookNLP/MuSciClaims",
            filename=corrected_path,
            repo_type="dataset"
        )
        
        # Get the filename from the path
        filename = os.path.basename(figure_path)
        destination = os.path.join(output_dir, filename)
        
        # Copy the file to paper_figures directory
        shutil.copy2(local_path, destination)
        
        success_count += 1
        print(f"[{success_count}] Downloaded: {filename}")
        
    except Exception as e:
        error_count += 1
        print(f"Error downloading {figure_path}: {e}")
print(f"Downloaded to: {os.path.abspath(output_dir)}")
