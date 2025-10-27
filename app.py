# app.py
import streamlit as st
from model import load_model, predict

st.title("🧠 Simple Image Classification App")
st.write("Upload an image and classify it using a pretrained ResNet18 model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")

    # Save temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    model = load_model()
    preds = predict(model, "temp.jpg")

    st.subheader("Top 5 Predictions:")
    for label, prob in preds:
        st.write(f"**{label}** — {prob:.4f}")
