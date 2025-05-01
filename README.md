This project presents a full-stack implementation of a Latent Diffusion Model (LDM) pipeline trained for image generation from text prompts. We focus on Approach 3, which fine-tunes only the U-Net component of Stable Diffusion using a custom dataset while leveraging pre-trained VAE and CLIP tokenizer/encoder components. The final result is integrated into a Gradio web application.

📌 Highlights
Model Type: Latent Diffusion (Stable Diffusion v1.5 backbone)

Fine-Tuning Strategy: Partial fine-tuning (U-Net only)

Dataset: Custom curated dataset (e.g., Butterflies subset)

Tech Stack: PyTorch, Hugging Face 🤗 Diffusers, Gradio

Deployment: Designed for Hugging Face Spaces, local Gradio app, or Streamlit

🚀 Features
Fine-tunes only the U-Net with frozen VAE and CLIP

Accelerated training with efficient resource usage (reduced GPU memory)

Generates high-quality images from text prompts

Includes training metrics: Loss curves, LPIPS, FID, CLIPScore

Gradio-based interactive UI for prompt-to-image inference

🧰 Tools & Libraries
Purpose	Tool/Library
Model Backbone	runwayml/stable-diffusion-v1-5
Training Framework	PyTorch, 🤗 Diffusers
Dataset Management	Hugging Face Datasets
Evaluation Metrics	LPIPS, FID, CLIPScore, PSNR, SSIM
UI Deployment	Gradio
Hardware Used	Google Colab A100

📁 Project Structure
bash
Copy
Edit
├── GenAI_final_project3.ipynb        # Main training and inference notebook
├── saved_unet/                       # Directory to save fine-tuned U-Net weights
├── app.py                            # Gradio app for prompt-to-image generation
├── requirements.txt                  # Required Python packages
└── README.md                         # You are here
🔧 Setup Instructions
Clone the repo

bash
Copy
Edit
git clone https://github.com/yourusername/latent-diffusion-webapp.git
cd latent-diffusion-webapp
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Launch the app

bash
Copy
Edit
python app.py
