import streamlit as st
from PIL import Image
import torch
from model import load_model, predict

st.set_page_config(page_title="Image Classification Demo", layout="centered")

st.title("🖼️ Image Classification with ResNet-18")
st.write("Upload a JPG or PNG image to see top-5 ImageNet predictions.")

# Load model once (cached)
@st.cache_resource
def get_model():
    return load_model()

model, preprocess, class_names = get_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running inference..."):
        predictions = predict(image, model, preprocess, class_names)

    st.subheader("Top-5 Predictions")
    for label, prob in predictions:
        st.write(f"**{label}**: {prob:.2%}")
