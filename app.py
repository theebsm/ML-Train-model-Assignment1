import streamlit as st
from fastai.vision.all import *
from PIL import Image
import io

# Cache model
@st.cache_resource
def load_model():
    learn = load_learner('model.pkl')
    return learn

st.title("🥦 Vegetable Classifier")
st.write("Upload a vegetable photo to identify it!")

# Load model
try:
    learn = load_model()
    st.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Convert to fastai PILImage ✅
    img_bytes = uploaded_file.getvalue()
    img_fastai = PILImage.create(io.BytesIO(img_bytes))

    # Predict
    with st.spinner("Identifying..."):
        pred, idx, probs = learn.predict(img_fastai)
        confidence = float(probs[idx]) * 100

    # Show result
    st.success(f"🎯 Prediction: **{pred}**")
    st.info(f"Confidence: **{confidence:.1f}%**")

    # Top 3
    st.write("### Top 3 predictions:")
    top3 = sorted(
        zip(learn.dls.vocab, probs),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    for cls, prob in top3:
        st.progress(float(prob))
        st.write(f"{cls}: {float(prob)*100:.1f}%")
