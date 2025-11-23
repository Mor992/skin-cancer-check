import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import requests
import tempfile
from PIL import Image

# ---------------------------------------------------------
# 1) DIRECT GOOGLE DRIVE DOWNLOAD (GUARANTEED WORKING)
# ---------------------------------------------------------
FILE_ID = "1uHgOzbvTY8hus4_ApzLlv7VO-Ye5uWpX"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

@st.cache_resource
def load_model():
    st.info("Downloading model from Google Drive...")

    session = requests.Session()
    response = session.get(DOWNLOAD_URL, stream=True)

    # handle Google Drive confirmation token
    def get_confirm(resp):
        for key, val in resp.cookies.items():
            if key.startswith("download_warning"):
                return val
        return None

    token = get_confirm(response)
    if token:
        response = session.get(f"{DOWNLOAD_URL}&confirm={token}", stream=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp:
        for chunk in response.iter_content(1024 * 1024):
            temp.write(chunk)
        model_path = temp.name

    st.info("Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model load FAILED: {e}")
        return None

model = load_model()

# ---------------------------------------------------------
# 2) Grad‑CAM function
# ---------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    backbone = model.get_layer("resnet50")
    last_conv = backbone.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    heatmap = np.zeros(conv_outputs.shape[:2])
    for i in range(pooled_grads.shape[0]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap

# ---------------------------------------------------------
# 3) UI
# ---------------------------------------------------------
st.title("🔥 Skin Cancer Detector + Grad‑CAM")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded and model is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image")

    img_resized = img.resize((224, 224))
    img_arr = np.array(img_resized) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr)
    class_idx = np.argmax(preds)

    st.subheader("Prediction")
    st.write(f"Class: **{class_idx}**")
    st.write(f"Confidence: **{preds[0][class_idx]:.4f}**")

    heatmap = make_gradcam_heatmap(img_arr, model)

    # overlay
    orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(heatmap_color, 0.4, orig, 0.6, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    st.subheader("Grad‑CAM")
    st.image(superimposed_rgb)
