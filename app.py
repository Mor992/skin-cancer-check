import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DRIVE_ID = "1lcgugQfz50PDNVFR6zxGZYuCJtm2IwfD"
MODEL_PATH = "final_resnet_model.keras"

st.title("Skin Lesion Classifier")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    # Download if missing
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model...")
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# -----------------------------
# IMAGE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    target_size = model.input_shape[1:3]
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, 0)

    # Predict
    class_names = [
        "Melanoma → Cancer (malignant)",
        "Melanocytic Nevus → Usually benign",
        "Basal Cell Carcinoma → Cancer",
        "Actinic Keratosis → Pre-cancerous"
    ]

    pred = model.predict(img_array)[0]
    idx = int(np.argmax(pred))
    confidence = float(pred[idx])

    st.subheader("Prediction Result")
    st.write(f"Top class: **{class_names[idx]}** ({confidence:.4f})")

    st.write("Full probabilities:")
    for name, p in zip(class_names, pred):
        st.write(f"{name}: {p:.4f}")

else:
    st.info("Upload an image to classify.")
