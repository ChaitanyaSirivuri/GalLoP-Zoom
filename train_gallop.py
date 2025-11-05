import os
import torch
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    LlavaNextForConditionalGeneration, # Added LLaVA
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PromptTuningConfig, PromptTuningInit
from PIL import Image
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train GalLoP (Global + Local Prompts) on MuSciClaims")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model ID")
    parser.add_argument("--data_dir", type=str, default="paper_figures", help="Directory with images")
    parser.add_argument("--output_dir", type=str, default="gallop_checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size") 
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate (Higher for Prompt Tuning)")
    return parser.parse_args()

def resolve_image_path(row, data_dir):
    figure_path = row['associated_figure_filepath']
    filename = os.path.basename(figure_path)
    if filename.startswith("jacs_data_10.1021_"):
        filename = filename.replace("jacs_data_10.1021_", "chem_10.1021_")
    path = os.path.join(data_dir, filename)
    return path

def prepare_dataset(model_id, data_dir):
    print("Loading dataset...")
    try:
        ds = load_dataset("StonyBrookNLP/MuSciClaims", split="train")
        print("Found train split.")
    except:
        print("Using test split (Warning: Training on test data)")
        ds = load_dataset("StonyBrookNLP/MuSciClaims", split="test")
    return ds

def train():
    args = get_args()
    
    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    # 2. Dynamic Model Loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    if "llava" in args.model_id.lower():
        print(f"Loading LLaVA-NeXT model: {args.model_id}")
        model_cls = LlavaNextForConditionalGeneration
    else:
        print(f"Loading Qwen model: {args.model_id}")
        model_cls = Qwen2_5_VLForConditionalGeneration
    
    model = model_cls.from_pretrained(
        args.model_id,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    # 3. Prompt Tuning Config (Soft Prompts)
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="Analyze accuracy:", # Short init
        tokenizer_name_or_path=args.model_id,
    )
    
    model = get_peft_model(model, peft_config)
    
    # Freeze logic
    for param in model.parameters():
        param.requires_grad = False
        
    # Enable gradients for prompt encoder
    if hasattr(model, "prompt_encoder"):
        for param in model.prompt_encoder.parameters():
            param.requires_grad = True
    
    model.print_trainable_parameters()
    
    # 4. Dataset
    ds = prepare_dataset(args.model_id, args.data_dir)
    def filter_images(x):
        return os.path.exists(resolve_image_path(x, args.data_dir))
    ds = ds.filter(filter_images)
    
    # 5. Data Collator (Handles both Qwen and LLaVA)
    def collate_fn(batch):
        texts = []
        images = []
        
        for row in batch:
            path = resolve_image_path(row, args.data_dir)
            image = Image.open(path).convert("RGB")
            images.append(image)
            
            caption = row.get('caption', '') or row.get('figure_caption', '')
            claim = row['claim_text']
            label = row['label_3class']
            
            # Format depends on model
            if "llava" in args.model_id.lower():
                # LLaVA Format: [INST] <image>\nPrompt [/INST] Response
                # Note: LLaVA-NeXT usually expects <image> token.
                prompt_text = f"[INST] <image>\nContext: {caption}\nClaim: {claim}\nVerify: [/INST] {label}"
                texts.append(prompt_text)
            else:
                # Qwen Format (ChatML)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": f"Context: {caption}\nClaim: {claim}\nVerify: "}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": label}]
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                texts.append(text)
            
        # Process inputs
        # LLaVA processor handles images differently (list of images)
        if "llava" in args.model_id.lower():
             inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        else:
             # Qwen
             inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
             
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs

    # 6. Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
        optim="paged_adamw_8bit" 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate_fn,
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    train()
