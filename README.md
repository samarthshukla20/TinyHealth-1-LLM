# 🩺 TinyHealth-1: Specialized LLM for Health Myth Checking

This project demonstrates a full-stack, end-to-end Machine Learning Operations (MLOps) workflow. The goal was to build a highly specialized Large Language Model (LLM) capable of fact-checking complex health claims, designed specifically to run on resource-constrained hardware.

The core achievement was successfully training and deploying a model that focuses **only** on medical queries.

---

## 🚀 Live Application & Deployment

The custom-trained model is hosted on Hugging Face Spaces and runs as a secure, dedicated API endpoint.

* **Live App Demo:** [Insert the direct link to your running Hugging Face Space URL here]
* **Model Repository:** [Insert the link to your Hugging Face Model Repo: `https://huggingface.co/samarthshukla/TinyHealth-1`]

---

## ✨ Project Highlights

* **Hardware Efficiency:** Successfully trained and merged the model using **QLoRA** on a **consumer-grade 4GB VRAM GPU**.
* **Domain Focus:** The model provides structured verdicts (Fact/Myth) and comprehensive analysis for health claims only.
* **Robust Filtering:** The model is highly accurate in rejecting non-health topics (e.g., coding, history, sports), which was achieved by balancing thousands of positive facts with over **75 explicit rejection examples**.

---

## 🛠️ Technical Stack & Workflow

This solution replaces expensive cloud APIs with a fully customizable, self-hosted LLM.

| Component | Tool / Library | Role in Project |
| :--- | :--- | :--- |
| **Base Model** | `TinyLlama/1.1B-Chat-v1.0` | Small, efficient base model selected to fit VRAM constraints. |
| **Fine-Tuning** | **PyTorch**, `trl` library, **QLoRA** | Framework and technique used for parameter-efficient training. |
| **Data Source** | **MedQuAD** + Custom Data | Factual base combined with custom instruction-tuning for filtering. |
| **API Backend** | **Flask** (Python) + `app.py` | Serves the model for inference in the Hugging Face Space. |
| **Hosting** | **Hugging Face Spaces** | Cloud platform used to host the model and run the Flask API on T4 GPU hardware. |
| **Frontend** | `index.html`, **Tailwind CSS** | The single-page application that calls the live API endpoint. |

---

## ⚙️ How to Replicate

The repository contains the full source code for the front-end, API, and training pipeline (`finetune_health.py`).

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/samarthshukla20/TinyHealth-1-LLM
    ```
2.  **Install Dependencies:**
    ```bash
    pip install torch transformers peft datasets pandas
    ```
3.  **Run Inference Locally:** Once you have the model files, you can test inference using the included Python script:
    ```bash
    python check_model.py "Drinking lemon water cures inflammation."
    ```
