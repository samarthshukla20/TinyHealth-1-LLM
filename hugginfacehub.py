import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError # <-- The Fix

# --- CONFIGURATION (Make sure these are correct) ---
# Replace 'samarthshukla' with your actual username if necessary
USERNAME = "samarthshukla" 
MODEL_NAME = "TinyHealth-1" 
REPO_ID = f"{USERNAME}/{MODEL_NAME}"
LOCAL_FOLDER = "./health_myth_checker_tinylama/final_merged_model" 

# --- 1. Initialize API and Check Local Folder ---
api = HfApi()

if not os.path.exists(LOCAL_FOLDER):
    print(f"FATAL ERROR: Local model folder '{LOCAL_FOLDER}' not found. Did you delete it?")
    print("Please ensure your trained model is in that directory.")
    exit()

# --- 2. Create the Repository (The Fix) ---
print(f"Attempting to create repository: {REPO_ID}")
try:
    # Use the create_repo function directly
    create_repo(
        repo_id=REPO_ID, 
        repo_type="model", 
        private=False, # Set to True if you want the repo to be private
        exist_ok=True # Allows the script to continue if the repo already exists
    )
    print("✅ Repository creation/check successful.")

except Exception as e:
    # Check for the specific error related to the repo existing or permissions
    if isinstance(e, RepositoryNotFoundError):
        print("Repo exists or permissions are wrong.")
    
    # Catch any exceptions related to login or token permissions
    print(f"\n❌ ERROR: Could not create repository. Ensure you are logged in with a token that has 'write' permissions.")
    print(f"Error details: {e}")
    exit()


# --- 3. Upload the Folder ---
print(f"Starting upload of files from '{LOCAL_FOLDER}'...")

try:
    api.upload_folder(
        folder_path=LOCAL_FOLDER,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Final merged {MODEL_NAME} model for health fact-checking."
    )
    print("\n=======================================================")
    print(f"✅ Upload Complete! Your model is live on Hugging Face:")
    print(f"   https://huggingface.co/{REPO_ID}")
    print("=======================================================")

except Exception as e:
    print(f"\n❌ ERROR during upload, even after repo creation.")
    print(f"Error details: {e}")