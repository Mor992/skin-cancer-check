# streamlit_gradcam_app.py
# Simple Streamlit app to load a Keras model, run predictions, and show Grad-CAM.
# MODEL_LOCAL_PATH should point to the model file in the environment.
# If the local file is not present, the app will attempt to download from Drive or allow upload.

import io
import os
import tempfile
from typing import List

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import load_model
import requests  # Added for downloading from URL

# --- CONFIG ---
# Local path to your model file (the environment path expected).
MODEL_LOCAL_PATH = "final_resnet_model.keras"
# Drive link for direct download (converted to direct download URL)
MODEL_DRIVE_URL = "https://drive.google.com/file/d/1kJWpQQlF-2Rtwj2xRmVtbDw-83cyjD3q/view?usp=sharing"

# Default class names (override in app UI if different)
DEFAULT_CLASS_NAMES = "benign,melanoma,other"

st.set_page_config(page_title="Grad-CAM Streamlit Demo", layout="wide")
st.title("Grad-CAM demo — upload image, predict, and explain")

# Utility: load model from path (can be local or URL)
@st.cache_resource
def load_keras_model(path: str):
    # load_model with compile=False to avoid requiring custom objects if needed
    return load_model(path, compile=False)

def get_model():
    if os.path.exists(MODEL_LOCAL_PATH):
        try:
            model = load_keras_model(MODEL_LOCAL_PATH)
            return model, MODEL_LOCAL_PATH
        except Exception as e:
            st.error(f"Failed to load model at {MODEL_LOCAL_PATH}: {e}")
            return None, None
    else:
        st.warning("Model file not found at the default path. Attempting to download from Drive link...")
        try:
            response = requests.get(MODEL_DRIVE_URL)
            if response.status_code == 200:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
                tmp.write(response.content)
                tmp.flush()
                tmp.close()
                model = load_keras_model(tmp.name)
                st.success("Model downloaded and loaded from Drive link.")
                return model, tmp.name
            else:
                raise Exception(f"Failed to download model from Drive: HTTP {response.status_code}")
        except Exception as e:
            st.error(f"Failed to load model from Drive: {e}")
            st.info("Please upload a Keras model file (.keras or .h5) manually.")
            uploaded = st.file_uploader("Upload a Keras model file (.keras or .h5)", type=["keras","h5"], key="model_upload")
            if uploaded is not None:
                # save to temp file then load
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
                tmp.write(uploaded.read())
                tmp.flush()
                tmp.close()
                try:
                    model = load_keras_model(tmp.name)
                    return model, tmp.name
                except Exception as e:
                    st.error(f"Failed to load uploaded model: {e}")
                    return None, None
        return None, None

def preprocess_pil(img: Image.Image, target_size: tuple):
    img = img.convert("RGB")
    img = img.resize(target_size[1:3])  # Assuming target_size is (batch, height, width, channels)
    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    # Basic preprocessing: scale to [0,1]
    arr = arr / 255.0
    return arr

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Adapted from Keras docs
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of class output w.r.t. conv layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Mean pooling over channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap

def overlay_heatmap(img: Image.Image, heatmap, alpha=0.4):
    import matplotlib.cm as cm
    import numpy as np

    heatmap = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    colored = colormap(heatmap)
    colored = np.uint8(colored[:, :, :3] * 255)
    colored_img = Image.fromarray(colored).resize(img.size)
    blended = Image.blend(img.convert("RGBA"), colored_img.convert("RGBA"), alpha)
    return blended

# --- Main app ---
model, model_path = get_model()

st.sidebar.header("Options")
class_names_input = st.sidebar.text_input("Class names (comma separated)", value=DEFAULT_CLASS_NAMES)
class_names = [c.strip() for c in class_names_input.split(",") if c.strip()]

if model is not None:
    st.sidebar.success(f"Model loaded from: {model_path}")
    # infer input shape
    try:
        input_shape = model.input_shape
        st.sidebar.write(f"Model input shape: {input_shape}")
    except Exception:
        input_shape = (None, 224, 224, 3)

    # get last convolutional layer automatically
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break
    if last_conv_layer_name is None:
        st.error("Couldn't automatically find a 4D conv layer in the model to use for Grad-CAM. Please ensure model has at least one Conv2D layer.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Input image", use_column_width=True)

        # preprocess
        img_array = preprocess_pil(img, input_shape)

        # predict
        preds = model.predict(img_array)
        if preds.ndim == 2 and preds.shape[1] > 1:
            probs = tf.nn.softmax(preds[0]).numpy()
        else:
            # binary or single-output, assume sigmoid
            probs = tf.nn.sigmoid(preds[0]).numpy()
            probs = np.array([1 - probs[0], probs[0]])  # Convert to [prob_class0, prob_class1]
        top_idx = int(np.argmax(probs))
        top_prob = float(np.max(probs))
        predicted_class = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"

        st.markdown(f"**Prediction:** {predicted_class}   —   **Confidence:** {top_prob:.3f}")

        # create grad-cam
        if last_conv_layer_name:
            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=top_idx)
                cam_img = overlay_heatmap(img, heatmap)
                st.image(cam_img, caption="Grad-CAM overlay", use_column_width=True)
            except Exception as e:
                st.error(f"Failed to compute Grad-CAM: {e}")

        # generate simple report based on predicted type
        st.subheader("Automated report (draft)")
        report_lines = []
        report_lines.append(f"Predicted: {predicted_class} (confidence {top_prob:.3f})")
        if predicted_class.lower() in ["melanoma", "cancer", "malignant"]:
            report_lines.append("Recommendation: Refer to a dermatologist/oncologist for confirmatory biopsy and further clinical assessment.")
            report_lines.append("Suggested next steps:")
            report_lines.append(" - Clinical examination by specialist")
            report_lines.append(" - Dermoscopic imaging")
            report_lines.append(" - Biopsy and histopathological analysis if indicated")
        else:
            report_lines.append("This prediction does not indicate malignancy. However, clinical correlation is required.")
            report_lines.append("Suggested next steps:")
            report_lines.append(" - Regular monitoring")
            report_lines.append(" - Consult a healthcare professional for any concerns")
        report_text = "\n".join(report_lines)
        st.text_area("Report", value=report_text, height=200)

        # allow download
        st.download_button("Download report as .txt", report_text, file_name="automated_report.txt")

else:
    st.stop()

# Footer note
st.markdown("---")
st.caption("Notes: This demo is for educational purposes. Do not use for clinical decision-making.")
