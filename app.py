import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile
import requests
import cv2
from PIL import Image
import io

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")

FILE_ID = "1uHgOzbvTY8hus4_ApzLlv7VO-Ye5uWpX"   # your model
CLASS_NAMES = ["benign", "malignant", "nevus", "other"]
LAST_CONV_LAYER = "conv5_block3_out"  # For ResNet50


# -----------------------------
# DOWNLOAD MODEL FROM DRIVE
# -----------------------------
def download_model_from_drive(file_id):
    """Returns the path of the downloaded .h5 file."""
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = requests.Session()
    response = session.get(URL, stream=True)

    # Detect virus scan confirmation
    def get_confirm_token(resp):
        for key, val in resp.cookies.items():
            if key.startswith("download_warning"):
                return val
        return None

    token = get_confirm_token(response)
    if token:
        URL = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
        response = session.get(URL, stream=True)

    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp:
        for chunk in response.iter_content(1024 * 1024):
            temp.write(chunk)
        return temp.name


@st.cache_resource
def load_model():
    st.info("Downloading model from Google Drive...")
    model_path = download_model_from_drive(FILE_ID)

    st.info("Loading model...")
    model = tf.keras.models.load_model(model_path)

    st.success("Model loaded successfully!")
    return model


model = load_model()


# -----------------------------
# GRAD‑CAM FUNCTIONS
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER):
    backbone = model.get_layer("resnet50")
    last_conv_layer = backbone.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.input], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0].numpy()
    pooled = pooled_grads.numpy()

    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i in range(len(pooled)):
        heatmap += pooled[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-6

    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed


# -----------------------------
# UI
# -----------------------------
st.title("🔥 Skin Lesion Classifier + Grad‑CAM")
st.write("Upload an image to see prediction & heatmap.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = pil_img.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Prediction
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    conf = float(preds[0][idx])

    st.subheader("📌 Prediction")
    st.write(f"**Class:** {CLASS_NAMES[idx]}")  
    st.write(f"**Confidence:** {conf:.4f}")

    # Grad‑CAM
    with st.spinner("Generating Grad‑CAM..."):
        heatmap = make_gradcam_heatmap(img_array, model)
        img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        overlay = overlay_heatmap(img_bgr, heatmap)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        st.subheader("🔥 Grad‑CAM Heatmap")
        st.image(overlay_rgb, use_column_width=True)

        # Download button
        is_ok, buf = cv2.imencode(".png", overlay)
        if is_ok:
            st.download_button(
                "⬇️ Download Heatmap",
                data=io.BytesIO(buf.tobytes()).getvalue(),
                file_name="gradcam.png",
                mime="image/png",
            )
