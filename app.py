import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import requests
import tempfile
from PIL import Image

st.set_page_config(page_title="Skin Classifier", layout="centered")

FILE_ID = "1uHgOzbvTY8hus4_ApzLlv7VO-Ye5uWpX"   # your model file ID
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# -------------------------------------------
# AUTO‑LOAD MODEL DIRECTLY FROM GOOGLE DRIVE
# -------------------------------------------
@st.cache_resource
def load_model_from_drive():
    st.info("Downloading model from Google Drive...")

    session = requests.Session()
    response = session.get(MODEL_URL, stream=True)

    # Handle Google Drive virus scan confirmation token
    def get_confirm_token(resp):
        for key, val in resp.cookies.items():
            if key.startswith("download_warning"):
                return val
        return None

    token = get_confirm_token(response)
    if token:
        download_url = MODEL_URL + "&confirm=" + token
        response = session.get(download_url, stream=True)

    # Save model to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                temp.write(chunk)
        model_path = temp.name

    st.info("Loading model into memory...")

    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# LOAD ON STARTUP
model = load_model_from_drive()

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------
uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

def preprocess(img):
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, 0)

# ---------------------------------------------------
# PREDICT + HEATMAP
# ---------------------------------------------------
if uploaded and model is not None:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, caption="Uploaded Image", use_column_width=True)

    arr = preprocess(pil)

    # Prediction
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    st.subheader("Prediction")
    st.write(f"Class: **{idx}**")
    st.write(f"Confidence: **{conf:.4f}**")

    # Optionally add GradCAM later
