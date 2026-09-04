import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Resolve path relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "health_myth_checker_tinylama", "final_merged_model")
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def check_claim_locally(claim_text):
    if not os.path.isdir(MODEL_DIR):
        print(f"\n❌ Error: Folder not found at '{MODEL_DIR}'")
        print("Check if the model was trained/merged and saved to this path.")
        return

    try:
        # 1. Load tokenizer and configure padding
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("--- Loading Model Weights into Memory ---")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    except Exception as e:
        print(f"\n❌ Error loading model components: {e}")
        return

    # 2. Format Prompt
    prompt = (
        "## Health Claim Checker\n\n"
        f"### Claim\n{claim_text}\n\n"
        "### Response\n"
    )

    # 3. Create Inference Pipeline
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        do_sample=True,          # Required when using temperature / top_k
        temperature=0.1,
        top_k=10,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False   # Returns only generated text after the prompt
    )

    # 4. Run Generation
    print("--- Running Inference ---")
    result = pipe(prompt)
    response_text = result[0]["generated_text"].strip()

    print("\n" + "=" * 50)
    print(f"Checking Claim: '{claim_text}'")
    print("=" * 50)
    print(response_text)
    print("=" * 50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python check_model.py "<Your health claim here>"')
        print('Example: python check_model.py "Does drinking lemon water cure all inflammation?"')
        sys.exit(1)

    claim = sys.argv[1]
    check_claim_locally(claim)