import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify

# --- CONFIGURATION ---
# IMPORTANT: This must match the repository name you uploaded:
FINETUNED_MODEL_REPO = "samarthshukla/TinyHealth-1" 
# Define the base model ID to load the verified tokenizer from it
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# The model will run on the first available device (GPU or CPU)
# The output shows it defaults to CPU because this is a free space, which is expected.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL LOADING ---
print(f"Loading model from: {FINETUNED_MODEL_REPO} to device: {DEVICE}")

try:
    # 1. CRITICAL FIX: Load the tokenizer using the correct variable name (BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # 2. Load the merged model (TinyLlama) from YOUR custom repo (FINETUNED_MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_MODEL_REPO,
        torch_dtype=torch.float16,
        device_map="auto", 
        trust_remote_code=True,
    )

    # 3. Create the text generation pipeline
    qa_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        # The 'device' argument is correctly omitted here
        max_new_tokens=400,
        temperature=0.1,
        repetition_penalty=1.1,
    )
    print("✅ Model and Pipeline loaded successfully!")

except Exception as e:
    print(f"❌ FATAL ERROR LOADING MODEL: {e}")
    # Raise the exception so the Space fails to start, which is better than serving a broken API
    raise e 

# --- FLASK API SETUP ---
app = Flask(__name__)

@app.route("/check_claim", methods=["POST"])
def check_claim():
    """Endpoint to receive a claim and return the structured verdict."""
    try:
        data = request.json
        claim = data.get("claim")

        if not claim:
            return jsonify({"error": "Missing 'claim' in request body."}), 400

        # Construct the exact prompt format the model was trained on
        prompt = (
            "You are a medical fact-checker. Verify if the following claim is a myth or fact.\n\n"
            f"Claim: {claim}\n"
            "Verdict (Myth / Fact / False):"
            )
        
        # Run inference
        result = qa_pipeline(prompt)
        
        # Extract and clean the generated text
        generated_text = result[0]['generated_text']
        response_text = generated_text.split('### Response\n')[-1].strip()
        
        # Return the raw, structured text to the web app
        return jsonify({"verdict": response_text}), 200

    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"error": "Internal inference error."}), 500

if __name__ == "__main__":
    # Run Flask server (usually handled by the Space environment, but included for local testing)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 7860)))