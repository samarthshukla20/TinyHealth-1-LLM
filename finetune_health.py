import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training # <-- Added preparation function
from trl import SFTTrainer

# --- 1. Configuration ---

# CRITICAL CHANGE: Switched to the smallest, ultra-VRAM-efficient TinyLlama model
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# The name of your custom fine-tuned model (will be the output folder)
new_model = "health_myth_checker_tinylama" 

# The name of your data file
dataset_name = "health_myth_data.jsonl" 

# --- 2. QLoRA and Quantization Setup ---

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=False,
)

# LoRA configuration 
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,                            
    bias="none",
    task_type="CAUSAL_LM",
    # Target all attention and feedforward layers
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
)

# --- 3. Loading the Model and Tokenizer ---

# Load the base model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto", 
    trust_remote_code=True,
)
# Disable cache for fine-tuning
model.config.use_cache = False 

# CRITICAL FIX: Prepare the model for QLoRA training
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# --- 4. Training Arguments ---

# Define hyperparameters for the training process
training_arguments = TrainingArguments(
    output_dir="./tinylama_results",
    num_train_epochs=3,                     
    per_device_train_batch_size=1,          
    gradient_accumulation_steps=4,          
    optim="paged_adamw_32bit",              
    save_steps=100,
    logging_steps=25,
    learning_rate=2e-4,                     
    weight_decay=0.001,
    fp16=True,                              
    bf16=False,                             
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)


dataset = load_dataset('json', data_files=dataset_name, split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments,
)

print("--- Starting Fine-Tuning ---")
trainer.train()

trainer.model.save_pretrained(new_model) 

print("--- Merging LoRA Adapter with Base Model for Final Output ---")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load the adapter weights and merge them with the base model weights
from peft import PeftModel
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload() 

# Save the final, merged model
output_merged_dir = os.path.join(new_model, "final_merged_model")
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)
tokenizer.save_pretrained(output_merged_dir)

print(f"\n✅ Training Complete! The final, merged model is saved to: {output_merged_dir}")