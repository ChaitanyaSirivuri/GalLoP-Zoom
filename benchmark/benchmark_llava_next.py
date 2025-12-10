"""
Benchmark script for LLaVA-Next (llava-v1.6-mistral-7b-hf) on MuSciClaims dataset.
"""

import os
import sys
import json
import torch
from tqdm import tqdm
from datetime import datetime
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.data_loader import load_dataset_with_images, get_image_for_sample
from benchmark.prompts import get_prompt_d, parse_decision, normalize_label
from benchmark.metrics import calculate_metrics, print_metrics, save_metrics_to_csv, save_predictions_to_csv

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def load_model():
    """Load LLaVA-Next model and processor."""
    print("Loading LLaVA-Next (llava-v1.6-mistral-7b-hf) model...")
    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    print("Model loaded successfully!")
    return model, processor


def run_inference(model, processor, image: Image.Image, prompt: str) -> str:
    """
    Run inference on a single sample.
    
    Args:
        model: The LLaVA-Next model
        processor: The processor
        image: PIL Image
        prompt: The text prompt
        
    Returns:
        Model's response string
    """
    # Prepare conversation in LLaVA format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    
    # Apply chat template
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    # Decode output
    response = processor.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from response (LLaVA includes it)
    # Find the last occurrence of the prompt or assistant marker
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    return response


def run_benchmark():
    """Run the full benchmark."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, processor = load_model()
    
    # Load dataset
    data = load_dataset_with_images()
    
    # Results storage
    results = []
    predictions = []
    labels = []
    
    # Run inference
    print(f"\nRunning benchmark on {len(data)} samples...")
    
    for sample in tqdm(data, desc="Processing"):
        # Get image
        image = get_image_for_sample(sample)
        
        if image is None:
            # Skip samples without images
            continue
        
        # Get prompt
        prompt = get_prompt_d(sample['claim_text'], sample['caption'])
        
        # Run inference
        try:
            response = run_inference(model, processor, image, prompt)
        except Exception as e:
            print(f"\nError on sample {sample['index']}: {e}")
            response = ""
        
        # Parse decision
        prediction = parse_decision(response)
        label = normalize_label(sample['label_3class'])
        
        # Store results
        predictions.append(prediction)
        labels.append(label)
        
        results.append({
            "index": sample['index'],
            "claim": sample['claim_text'],
            "caption": sample['caption'],
            "label": label,
            "prediction": prediction,
            "correct": prediction == label,
            "image_path": sample['image_path'],
            "panels": sample['associated_figure_panels'],
            "raw_response": response,
        })
        
        # Close image to save memory
        image.close()
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    
    # Print results
    print_metrics(metrics, "LLaVA-Next-Mistral-7B")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metrics_path = os.path.join(output_dir, f"llava_next_metrics_{timestamp}.csv")
    save_metrics_to_csv(metrics, metrics_path, "LLaVA-Next-Mistral-7B")
    
    predictions_path = os.path.join(output_dir, f"llava_next_predictions_{timestamp}.csv")
    save_predictions_to_csv(results, predictions_path)
    
    # Save full results as JSON
    json_path = os.path.join(output_dir, f"llava_next_results_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "LLaVA-Next-Mistral-7B",
            "timestamp": timestamp,
            "metrics": metrics,
            "total_samples": len(results),
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    
    return metrics, results


if __name__ == "__main__":
    run_benchmark()
