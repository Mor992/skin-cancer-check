import streamlit as st
import requests
import tensorflow as tf
import numpy as np
import cv2
import io
import tempfile

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# =========================================================
#                  GOOGLE DRIVE DOWNLOAD
# =========================================================
def download_file_from_google_drive(file_id: str) -> str:
    """Download a large file from Google Drive and return its local filepath."""
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    # Look for Drive's virus-scan confirmation token
    def get_confirm_token(resp):
        for key, val in resp.cookies.items():
            if key.startswith("download_warning"):
                return val
        return None

    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                temp.write(chunk)
        return temp.name



# =========================================================
#                LOAD MODEL (CACHED)
# =========================================================
@st.cache_resource
def load_model_from_drive(file_id: str):
    """Download and load a .h5 model from Google Drive."""
    try:
        st.info("Downloading model from Google Drive…")
        model_path = download_file_from_google_drive(file_id)

        st.info("Loading model into memory…")
        model = tf.keras.models.load_model(model_path)

        st.success("Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None



# =========================================================
#                    GRAD‑CAM
# =========================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    """Generate Grad‑CAM heatmap for an input image array."""
    try:
        backbone = model.get_layer("resnet50")
    except:
        raise ValueError("Model does not contain a 'resnet50' layer.")

    last_conv_layer = backbone.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs[0].numpy()

    heatmap = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i in range(len(pooled_grads)):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    return heatmap



# =========================================================
#                HEATMAP OVERLAY
# =========================================================
def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(heatmap_color, alpha, orig_bgr, 1 - alpha, 0)
    return blended



# =========================================================
#                IMAGE PREPROCESSING
# =========================================================
def preprocess_for_model(pil_img, target_size=(224, 224)):
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img)

    # Convert to float32 and normalize
    arr = arr.astype("float32")
    arr = preprocess_input(arr)

    return np.expand_dims(arr, axis=0)



# =========================================================
#                        UI
# =========================================================

st.title("🔥 Skin Lesion Classifier + Grad‑CAM")
st.write("Upload an image, load your model from Google Drive, and visualize Grad‑CAM.")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🔗 Load Model")
    file_id_input = st.text_input("Google Drive File ID", value="")
    last_conv_layer_name = st.text_input("Last Conv Layer", "conv5_block3_out")

    load_button = st.button("Load model")

model = None
if load_button:
    if not file_id_input.strip():
        st.error("Please enter a valid Google Drive File ID.")
    else:
        with st.spinner("Loading model…"):
            model = load_model_from_drive(file_id_input)



# =========================================================
#                    IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    if model is None:
        st.warning("Model not loaded. Please load a model first.")
        st.stop()

    # ---------------- Prediction ----------------
    with st.spinner("Running prediction…"):
        try:
            processed = preprocess_for_model(pil_img)
            preds = model.predict(processed)
            pred_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))

            CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3"]
            predicted_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"

            st.subheader("📌 Prediction")
            st.write(f"**Class:** {predicted_label}")
            st.write(f"**Confidence:** {confidence:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()


    # ---------------- Grad‑CAM ----------------
    with st.spinner("Generating Grad‑CAM…"):
        try:
            processed_for_cam = preprocess_for_model(pil_img)
            heatmap = make_gradcam_heatmap(processed_for_cam, model, last_conv_layer_name)

            orig_bgr = cv2.cvtColor(np.array(pil_img.resize((224, 224))), cv2.COLOR_RGB2BGR)
            overlay = overlay_heatmap_on_image(orig_bgr, heatmap)

            col1, col2 = st.columns(2)
            col1.image(pil_img, caption="Original", use_column_width=True)
            col2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Grad‑CAM", use_column_width=True)

            # Download button
            ok, png = cv2.imencode(".png", overlay)
            if ok:
                st.download_button(
                    "⬇️ Download Grad‑CAM Image",
                    data=png.tobytes(),
                    file_name="gradcam.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Grad‑CAM failed: {e}")
