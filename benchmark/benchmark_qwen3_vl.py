"""
Benchmark script for Qwen3-VL-8B-Instruct on MuSciClaims dataset.
"""

import os
import sys
import json
import torch
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.data_loader import load_dataset_with_images, get_image_for_sample
from benchmark.prompts import get_prompt_d, parse_decision, normalize_label
from benchmark.metrics import calculate_metrics, print_metrics, save_metrics_to_csv, save_predictions_to_csv

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image


def load_model():
    """Load Qwen3-VL model and processor."""
    print("Loading Qwen3-VL-8B-Instruct model...")
    
    # Note: Qwen3-VL may require different class depending on transformers version
    # Try Qwen3VLMoeForConditionalGeneration first, fallback to Qwen2VLForConditionalGeneration
    try:
        from transformers import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype="auto",
            device_map="auto"
        )
    except ImportError:
        print("Qwen3VLMoeForConditionalGeneration not available, trying alternative...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    
    print("Model loaded successfully!")
    return model, processor


def run_inference(model, processor, image: Image.Image, prompt: str) -> str:
    """
    Run inference on a single sample.
    
    Args:
        model: The Qwen3-VL model
        processor: The processor
        image: PIL Image
        prompt: The text prompt
        
    Returns:
        Model's response string
    """
    # Prepare messages in Qwen format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Process inputs using apply_chat_template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    print(f"Generated IDs: {generated_ids}")
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print(f"Trimmed Generated IDs: {generated_ids_trimmed}")
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"Output Text: {output_text}")
    return output_text[0] if output_text else ""


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
    print_metrics(metrics, "Qwen3-VL-8B-Instruct")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metrics_path = os.path.join(output_dir, f"qwen3_vl_metrics_{timestamp}.csv")
    save_metrics_to_csv(metrics, metrics_path, "Qwen3-VL-8B-Instruct")
    
    predictions_path = os.path.join(output_dir, f"qwen3_vl_predictions_{timestamp}.csv")
    save_predictions_to_csv(results, predictions_path)
    
    # Save full results as JSON
    json_path = os.path.join(output_dir, f"qwen3_vl_results_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "Qwen3-VL-8B-Instruct",
            "timestamp": timestamp,
            "metrics": metrics,
            "total_samples": len(results),
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    
    return metrics, results


if __name__ == "__main__":
    run_benchmark()
