# app.py
import os
import io
import zipfile
import tempfile
import gdown
from typing import Optional

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# -------------------------
# Config / Change as needed
# -------------------------
DRIVE_FILE_ID = "1csR51feB60Uvzh3Qp4iMYuV9C0EkSMy_"  # change if needed
MODEL_LOCAL_NAME = "my_model.keras"                 # final expected local filename
MODEL_ZIP_NAME = "model_download.zip"               # used only if file is actually a zip
MODEL_IS_ZIP = False  # <-- set True ONLY if you actually zipped your model into a .zip on Drive

# Example class names - replace with your real labels/order if different
CLASS_NAMES = ["Melanoma", "Nevus", "Seborrheic Keratosis"]

# Small informational reports (non-diagnostic)
CLASS_REPORTS = {
    "Melanoma": "Melanoma is a serious form of skin cancer. Early clinical examination is recommended.",
    "Nevus": "A nevus (mole) is often benign, but changes in shape/color/size may require medical review.",
    "Seborrheic Keratosis": "A common, benign skin growth. Usually harmless but see a clinician for worrisome changes."
}

st.set_page_config(page_title="Skin Lesion Classifier (ResNet50)", layout="centered")

# -------------------------
# Utility Functions
# -------------------------
def download_model_from_drive(drive_id: str, dest: str, zip_dest: str = None, is_zip: bool = False):
    """Download model file from Google Drive (direct download)."""
    url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    if is_zip:
        if zip_dest is None:
            raise ValueError("zip_dest must be provided when is_zip=True")
        gdown.download(url, zip_dest, quiet=False)
        return zip_dest
    else:
        gdown.download(url, dest, quiet=False)
        return dest

def safe_load_model(path: str):
    """Load a Keras model with a friendly error message."""
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error("Failed to load model. See logs for details.")
        # Show a helpful hint to the user in app
        st.write("Model loading error:", e)
        raise

def preprocess_image_pil(pil_img: Image.Image, target_size=(224,224)):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def overlay_heatmap_on_image(original_image_np, heatmap, alpha=0.5):
    # original_image_np: HxWx3 uint8 RGB (0-255)
    hmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
    hmap = np.uint8(255 * hmap)
    hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)  # BGR
    hmap_color = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    overlay = (original_image_np.astype("float32") * (1 - alpha) + hmap_color.astype("float32") * alpha)
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    return overlay

def get_last_conv_layer(model: tf.keras.Model) -> Optional[str]:
    """
    Find the last convolutional layer name in the model.
    Looks for layers containing 'conv' and 4D output shape.
    """
    last_conv = None
    for layer in model.layers:
        if hasattr(layer, "output_shape"):
            shp = layer.output_shape
            # check for conv-like 4D tensor
            if isinstance(shp, tuple) and len(shp) == 4:
                name = layer.name.lower()
                if "conv" in name or "conv2d" in name:
                    last_conv = layer.name
    return last_conv

# -------------------------
# Grad-CAM++ (from notebook, fixed)
# -------------------------
def grad_cam_plus_plus(model, image, class_idx, layer_name):
    """
    Robust Grad-CAM++ implementation that handles cases where model.output
    might be a list/tuple. Returns a heatmap (HxW) with values in [0,1].
    """
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)

        # If predictions is a list/tuple (e.g., model.output = [tensor]), use the first element
        if isinstance(predictions, (list, tuple)):
            # prefer the first element that looks like a tensor with batch dim
            # fall back to predictions[0] if unsure
            preds_candidate = None
            for p in predictions:
                try:
                    # try to convert to tensor and check rank
                    p_tensor = tf.convert_to_tensor(p)
                    if p_tensor.shape.rank is not None and p_tensor.shape.rank >= 1:
                        preds_candidate = p_tensor
                        break
                except Exception:
                    continue
            if preds_candidate is None:
                # last resort
                preds_candidate = tf.convert_to_tensor(predictions[0])
            predictions = preds_candidate
        else:
            predictions = tf.convert_to_tensor(predictions)

        # Ensure predictions has a batch dimension
        # Now safe to index: predictions[0, class_idx]
        loss = predictions[0, class_idx]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)                    # shape (1, h, w, channels)
    grads_sq = tf.square(grads)
    grads_cu = grads_sq * grads

    alpha_num = grads_sq
    alpha_denom = 2 * grads_sq + tf.reduce_sum(conv_outputs * grads_cu, axis=(1,2), keepdims=True)
    alpha_denom = tf.where(alpha_denom != 0, alpha_denom, tf.ones_like(alpha_denom))

    alphas = alpha_num / alpha_denom

    weights = tf.reduce_sum(alphas * tf.maximum(grads, 0), axis=(1,2))
    weights = tf.reshape(weights, [-1,1,1,tf.shape(conv_outputs)[-1]])

    grad_cam_output = tf.reduce_sum(weights * conv_outputs, axis=-1)  # shape (1, h, w)
    heatmap = tf.maximum(grad_cam_output, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    heatmap = tf.squeeze(heatmap)  # shape (h, w)
    return heatmap.numpy()

# -------------------------
# Model download & load
# -------------------------
@st.cache_resource(show_spinner=False)
def get_model(download_if_missing: bool = True):
    # If model exists in repo, use it
    if os.path.exists(MODEL_LOCAL_NAME):
        model_path = MODEL_LOCAL_NAME
    else:
        if not download_if_missing:
            raise FileNotFoundError(f"{MODEL_LOCAL_NAME} not found.")
        st.info("Downloading model from Google Drive...")
        if MODEL_IS_ZIP:
            downloaded = download_model_from_drive(DRIVE_FILE_ID, MODEL_LOCAL_NAME, zip_dest=MODEL_ZIP_NAME, is_zip=True)
            # extract
            with zipfile.ZipFile(downloaded, "r") as z:
                z.extractall(".")
            # assume the zip contained MODEL_LOCAL_NAME or a folder; try both
            if os.path.exists(MODEL_LOCAL_NAME):
                model_path = MODEL_LOCAL_NAME
            else:
                # try to find a .keras or folder
                candidates = [f for f in os.listdir(".") if f.endswith(".keras") or os.path.isdir(f)]
                if candidates:
                    model_path = candidates[0]
                else:
                    raise FileNotFoundError("Could not locate model inside the extracted zip.")
        else:
            downloaded = download_model_from_drive(DRIVE_FILE_ID, MODEL_LOCAL_NAME, is_zip=False)
            model_path = MODEL_LOCAL_NAME

    # Load model - wrap with try for friendly Streamlit error
    model = safe_load_model(model_path)
    return model

# -------------------------
# Streamlit UI
# -------------------------
st.title("Skin Lesion Classifier (ResNet50)")
st.write("Upload an image to classify it into skin lesion categories.")

# Model load button / show model status
with st.spinner("Preparing model..."):
    try:
        model = get_model()
        st.success("Model loaded.")
    except Exception as e:
        st.error("Model could not be loaded. Check logs or model file.")
        st.stop()

# auto-detect last conv layer (fallback)
default_layer = "conv5_block3_3_conv"  # your notebook's layer (preferred if present)
detected_last_conv = get_last_conv_layer(model)
if detected_last_conv is None:
    detected_last_conv = default_layer  # fallback, may still error if not present

st.sidebar.header("Grad-CAM Settings")
layer_name = st.sidebar.text_input("Conv layer name for Grad-CAM (leave to auto-detect)", value=detected_last_conv)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM++", value=True)

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    try:
        image_data = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        st.error("Could not read the uploaded image.")
        st.stop()

    st.image(pil_img, caption="Uploaded Image", width=350)

    # Preprocess
    input_img = preprocess_image_pil(pil_img, target_size=(224,224))

    # Prediction
    preds = model.predict(input_img)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)
    pred_idx = int(np.argmax(preds[0]))
    pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
    pred_prob = float(preds[0, pred_idx])

    st.subheader("Prediction")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Probability:** {pred_prob:.4f}")

    st.subheader("Class Probabilities")
    for i, name in enumerate(CLASS_NAMES):
        prob = float(preds[0, i]) if i < preds.shape[1] else 0.0
        st.write(f"{name}: {prob:.4f}")

    # Grad-CAM generation
    if show_gradcam:
        st.subheader("Grad-CAM++ Visualization")
        # ensure layer exists; try fallback auto-detect
        try:
            _ = model.get_layer(layer_name)
        except Exception:
            # try auto-detect
            auto = get_last_conv_layer(model)
            if auto:
                st.warning(f"Layer '{layer_name}' not found. Using detected conv layer '{auto}' instead.")
                layer_name = auto
            else:
                st.error("Could not find a convolutional layer for Grad-CAM. Skipping Grad-CAM.")
                layer_name = None

        if layer_name:
            try:
                heatmap = grad_cam_plus_plus(model, input_img.astype("float32"), pred_idx, layer_name)
                # overlay on original (use original uploaded image resized to 224-> use original size)
                orig_np = np.array(pil_img.convert("RGB"))
                overlay = overlay_heatmap_on_image(orig_np, heatmap, alpha=0.5)
                st.image(overlay, caption="Grad-CAM++ overlay", use_column_width=False)
            except Exception as e:
                st.error("Grad-CAM generation failed. See logs for details.")
                st.write(e)

    # Small medical-style (non-diagnostic) report
    st.subheader("Information About Prediction")
    st.info(CLASS_REPORTS.get(pred_class, "No report available for this class."))

    # Option: download report as simple text file
    if st.button("Download simple report (.txt)"):
        txt = f"Prediction report\nClass: {pred_class}\nProbability: {pred_prob:.4f}\n\nNotes:\n{CLASS_REPORTS.get(pred_class, '')}\n"
        st.download_button("Click to download", data=txt, file_name="prediction_report.txt", mime="text/plain")

else:
    st.info("Please upload an image to classify.")
