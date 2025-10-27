# image-classification-app


# 🧠 Simple Image Classification App

A lightweight image classification demo using **PyTorch** and **Streamlit**, built with a pretrained **ResNet18** model from TorchVision.

## 🚀 Features
- Upload any `.jpg`, `.jpeg`, or `.png` image
- Classify it instantly using a pretrained CNN
- Shows top-5 ImageNet predictions with probabilities
- Easy to extend with custom models or datasets

## 🧩 Tech Stack
- **PyTorch** — deep learning framework  
- **TorchVision** — pretrained models and transforms  
- **Streamlit** — web app interface  
- **Pillow / Matplotlib** — image handling  

## 🛠️ Installation

```bash
git clone https://github.com/<your-username>/image-classification-app.git
cd image-classification-app
pip install -r requirements.txt
streamlit run app.py
