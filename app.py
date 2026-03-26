import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Load model
learn = load_learner('model.pkl')

st.title("🥦 Vegetable Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    pred_class, pred_idx, probs = learn.predict(img)
    
    st.write(f"### Prediction: {pred_class}")
    st.write(f"### Confidence: {probs[pred_idx]:.4f}")