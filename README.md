# 🧠 Latent Diffusion Image Generation Web App

This project presents a full-stack implementation of a **Latent Diffusion Model** (LDM) pipeline trained for image generation from text prompts. We focus on **Approach 3**, which fine-tunes only the **U-Net** component of Stable Diffusion using a custom dataset while leveraging pre-trained **VAE** and **CLIP tokenizer/encoder** components. The final result is integrated into a Gradio web application.

## 📌 Highlights

- **Model Type**: Latent Diffusion (Stable Diffusion v1.5 backbone)
- **Fine-Tuning Strategy**: Partial fine-tuning (U-Net only)
- **Dataset**: Custom curated dataset (e.g., Butterflies subset)
- **Tech Stack**: PyTorch, Hugging Face 🤗 Diffusers, Gradio
- **Deployment**: Designed for Hugging Face Spaces, local Gradio app, or Streamlit

## 🚀 Features

- Fine-tunes only the **U-Net** with **frozen VAE and CLIP**
- Accelerated training with efficient resource usage (reduced GPU memory)
- Generates high-quality images from text prompts
- Includes training metrics: **Loss curves**, **LPIPS**, **FID**, **CLIPScore**
- Gradio-based interactive UI for prompt-to-image inference

## 🧰 Tools & Libraries

| Purpose             | Tool/Library                    |
|---------------------|----------------------------------|
| Model Backbone      | `runwayml/stable-diffusion-v1-5` |
| Training Framework  | PyTorch, 🤗 Diffusers            |
| Dataset Management  | Hugging Face Datasets           |
| Evaluation Metrics  | LPIPS, FID, CLIPScore, PSNR, SSIM |
| UI Deployment       | Gradio                          |
| Hardware Used       | Google Colab A100               |

## 📁 Project Structure

```
├── GenAI_final_project3.ipynb        # Main training and inference notebook
├── saved_unet/                       # Directory to save fine-tuned U-Net weights
├── app.py                            # Gradio app for prompt-to-image generation
├── requirements.txt                  # Required Python packages
└── README.md                         # You are here
```

## 🔧 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/latent-diffusion-webapp.git
   cd latent-diffusion-webapp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app**
   ```bash
   python app.py
   ```

## 🧪 Evaluation Metrics

| Metric      | Purpose                                  |
|-------------|------------------------------------------|
| LPIPS       | Perceptual image similarity              |
| FID         | Frechet Inception Distance               |
| CLIP Score  | Text-image alignment measure             |
| PSNR/SSIM   | Low-level image fidelity                 |

## 🧠 Training Strategy – Approach 3

- Only the U-Net is fine-tuned
- Pre-trained VAE encoder/decoder and CLIP tokenizer remain frozen
- Training on 512×512 images using a lightweight butterfly dataset
- Reduced training cost and overfitting risks

## 📷 Example Outputs

| Prompt                        | Output Image (After Fine-Tuning)     |
|------------------------------|--------------------------------------|
| `"A majestic butterfly"`     | ![output](samples/butterfly1.png)   |
| `"Butterfly on a flower"`    | ![output](samples/butterfly2.png)   |

## 📦 Future Improvements

- Experiment with **LoRA** or **PEFT** methods for even more efficient fine-tuning
- Add **Streamlit** support for advanced UI controls
- Integrate **dataset upload** for user-specific fine-tuning

## 🧑‍💻 Authors

- Sathwik Reddy Chelemela – [Portfolio](https://chelemelasathwik.me)
- Northeastern University, GenAI & Deep Models, Spring 2025
