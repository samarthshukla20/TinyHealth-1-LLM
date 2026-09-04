# 🩺 TinyHealth-1: Specialized LLM for Health Myth Checking

This project showcases a complete Machine Learning Operations (MLOps) workflow for building a specialized, resource-efficient Large Language Model (LLM) designed to combat health misinformation.

The model was fine-tuned on a 4GB VRAM GPU to strictly filter non-medical queries and provide structured verdicts (Fact/Myth) for health claims.

---

## 🚀 Live Application & Model

The custom-trained model is hosted on Hugging Face Spaces and runs as a live API endpoint.

* **Live App Demo:** https://tinyhealthai.vercel.app/
* **Model Repository:**  https://huggingface.co/samarthshukla/TinyHealth-1

---

## ⚙️ Local Replication Guide (Run the Model on Your PC)

This guide directs users on how to recreate the specialized model and run local inference using your GPU.

### Step 1: Prerequisites

You must have the following software installed:

* **Python 3.9+**
* **NVIDIA GPU with CUDA:** Required for fast training and inference.

### Step 2: Set Up the Environment

1.  **Clone the repository and navigate into the folder:**
    ```bash
    git clone https://github.com/samarthshukla20/TinyHealth-1-LLM
    cd TinyHealth-1-LLM
    ```
2.  **Install Essential Libraries:** Install the core deep learning and ML frameworks used for training and inference.
    ```bash
    pip install torch transformers peft accelerate bitsandbytes trl datasets pandas
    ```

### Step 3: Train the Specialized Model

The training process uses the optimized **QLoRA** technique to adapt the TinyLlama model efficiently.

1.  **Check Data:** Ensure the training data (`health_myth_data.jsonl`) is in the main directory.
2.  **Start Training:** Execute the training script. This process will download the base model and run for several hours (4–8 hours, depending on GPU speed).
    ```bash
    python finetune_health.py
    ```

### Step 4: Run Local Inference (Check the Model)

Once the training completes and the final model files are saved into the `./health_myth_checker_tinylama/final_merged_model` folder, you can test its performance locally.

1.  **Execute the Inference Script:** Run the following command with a test claim enclosed in quotes:
    ```bash
    python check_model.py "Drinking a tablespoon of apple cider vinegar every morning burns belly fat."
    ```
2.  **Expected Output:** The model will load onto your GPU and return the structured verdict:
    ```
    ==================================================
    Checking Claim: 'Drinking a tablespoon...'
    ==================================================
    **Verdict: Myth**
    **Analysis:** [Detailed Explanation]
    **Recommendation:** [Safe Advice]
    ==================================================
    ```

---

## 🔎 Explanation of Deployment Files

| File Name | Role in the Project |
| :--- | :--- |
| **`finetune_health.py`** | The complete script used to **train the model** (Step 3). |
| **`check_model.py`** | The Python script used to **test the model locally** (Step 4). |
| **`app.py`** | The **Flask API server** code. This is the **live backend** running on Hugging Face Spaces that receives requests from the front-end website (`index.html`). |
| **`requirements.txt`** | The list of Python libraries the **Hugging Face Space** needs to install to run `app.py`. |
| **`index.html`** | The **Frontend** (website) that the user interacts with. It contains the JavaScript logic to call the live API endpoint defined in `app.py`. |
