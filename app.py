# app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os, json
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer

st.set_page_config(
    page_title="Latent Diffusion Demo",
    layout="centered",
)

@st.cache_resource(show_spinner=False)
def load_pipeline(model_dir: str):
    """
    Load your fine-tuned Stable Diffusion pipeline using custom components.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine the base model name from your fine-tuned UNet config
    unet_config_path = os.path.join(model_dir, "unet", "config.json")
    with open(unet_config_path, "r") as f:
        unet_config = json.load(f)
    base_model = unet_config.get("_name_or_path")
    if base_model is None:
        raise ValueError(f"Missing _name_or_path in {unet_config_path}")

    # Load the original pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )

    # Load fine-tuned UNet weights from safetensors file in the 'unet' subfolder
    try:
        from safetensors.torch import load_file as load_safetensors
    except ImportError:
        raise ImportError("Please install the 'safetensors' library to load UNet weights.")
    safetensors_path = os.path.join(model_dir, "unet", "diffusion_pytorch_model-001.safetensors")
    if not os.path.isfile(safetensors_path):
        raise FileNotFoundError(f"Cannot find UNet weight file at {safetensors_path}")
    state_dict = load_safetensors(safetensors_path, device=device)
    pipe.unet.load_state_dict(state_dict, strict=False)

    # Override the scheduler config
    scheduler = DDPMScheduler.from_pretrained(
        model_dir,
        subfolder="scheduler",
        local_files_only=True,
    )
    pipe.scheduler = scheduler

    # Override the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(model_dir, "tokenizer"),
        local_files_only=True,
    )
    pipe.tokenizer = tokenizer

    if device == "cuda":
        pipe = pipe.to(device)

    return pipe

# ==== CONFIGURE YOUR MODEL PATH HERE ====
MODEL_DIR = "model"    # or "./model" if app.py is next to the `model/` folder

# load once and cache
pipe = load_pipeline(MODEL_DIR)

# ==== UI LAYOUT ====
st.title("🎨 Latent Diffusion Text-to-Image")

prompt = st.text_input(
    "Enter your prompt",
    value="A colorful butterfly sitting on a flower at sunset"
)

col1, col2 = st.columns(2)
with col1:
    steps = st.slider(
        "Inference steps",
        min_value=10,
        max_value=100,
        value=50,
        step=5
    )
with col2:
    guidance = st.slider(
        "Guidance scale",
        min_value=1.0,
        max_value=15.0,
        value=7.5,
        step=0.5
    )

if st.button("Generate"):
    with st.spinner("Generating…"):
        output = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        image = output.images[0]
        st.image(
            image,
            caption=prompt,
            use_column_width=True
        )
