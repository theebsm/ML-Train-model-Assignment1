
import streamlit as st
from fastai.vision.all import *
from PIL import Image

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model():
    learn = load_learner('model.pkl')
    learn.dls.num_workers = 0  # 🔥 Fix for Streamlit crashes
    return learn

st.title("🥦 Vegetable Classifier")
st.write("Upload a vegetable photo to identify it!")

# -------------------------
# Load model safely
# -------------------------
try:
    learn = load_model()
    st.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------
# Upload image
# -------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        # ✅ Open and clean image properly
        img = Image.open(uploaded_file).convert("RGB")

        # Show image
        st.image(img, caption="Uploaded Image", width=300)

        # ✅ Correct FastAI conversion (IMPORTANT FIX)
        img_fastai = PILImage.create(img)

        # -------------------------
        # Predict
        # -------------------------
        with st.spinner("Identifying..."):
            pred, idx, probs = learn.predict(img_fastai)
            confidence = float(probs[idx]) * 100

        # -------------------------
        # Show results
        # -------------------------
        st.success(f"🎯 Prediction: **{pred}**")
        st.info(f"Confidence: **{confidence:.1f}%**")

        # Top 3 predictions
        st.write("### Top 3 predictions:")
        top3 = sorted(
            zip(learn.dls.vocab, probs),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        for cls, prob in top3:
            st.progress(float(prob))
            st.write(f"{cls}: {float(prob)*100:.1f}%")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.error(str(e))
