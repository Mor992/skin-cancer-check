# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import tempfile
import requests
import io
from PIL import Image

st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")

# ---------- Helpers: download from Google Drive (handles confirmation token) ----------
def download_file_from_google_drive(file_id):
    """Download a file from google drive and return path to temp file.
       Handles large-file confirm token flow.
    """
    base_url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(base_url, params={"id": file_id}, stream=True)
    token = None

    # if response content is a small HTML page, Google may require confirm token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        response = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True)

    # write to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    with tmp as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    return tmp.name

# ---------- Load model (cached) ----------
@st.cache_resource
def load_model_from_drive(file_id: str):
    try:
        st.info("Downloading model from Google Drive...")
        model_path = download_file_from_google_drive(file_id)
        st.info("Loading model into memory...")
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ---------- Grad-CAM core ----------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    """Create Grad-CAM heatmap for a single preprocessed image array.
       img_array shape: (1, H, W, 3) and should be preprocessed as model expects.
    """
    # Get internal ResNet backbone
    try:
        backbone = model.get_layer("resnet50")
    except Exception:
        raise ValueError("Model does not contain a 'resnet50' layer. Adjust last_conv_layer_name accordingly.")

    # Get last conv layer from backbone
    last_conv = backbone.get_layer(last_conv_layer_name)

    # Build model mapping input -> (last_conv_output, final_prediction)
    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    # Weight feature maps
    heatmap = np.zeros(shape=conv_outputs.shape[0:2], dtype=np.float32)
    for i in range(pooled_grads.shape[0]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    return heatmap  # shape (h, w) normalized 0..1

# ---------- Overlay heatmap ----------
def overlay_heatmap_on_image(orig_img_bgr, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    hmap = cv2.resize(heatmap, (orig_img_bgr.shape[1], orig_img_bgr.shape[0]))
    hmap = np.uint8(255 * hmap)
    heatmap_color = cv2.applyColorMap(hmap, colormap)
    superimposed = cv2.addWeighted(heatmap_color, alpha, orig_img_bgr, 1 - alpha, 0)
    return superimposed

# ---------- Preprocess single image ----------
def preprocess_for_model(pil_img, target_size=(224, 224)):
    img = pil_img.resize(target_size)
    arr = np.array(img).astype("float32")
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr * 255.0 if arr.max() <= 1 else arr)  # ensure proper range for preprocess_input
    return arr

# ---------- UI ----------
st.title("🔥 Skin Lesion Classifier + Grad-CAM")
st.write("Upload an image, get a prediction report, and download the Grad‑CAM heatmap.")

with st.sidebar:
    st.header("Model source")
    st.write("Provide the Google Drive file id for your `.h5` model (shared as 'Anyone with link can view').")
    file_id_input = st.text_input("Drive File ID", value="")
    if not file_id_input:
        st.info("Paste your Drive file ID (example: 1uHg... ).")

    st.markdown("---")
    st.write("Advanced:")
    last_conv_layer_name = st.text_input("Last conv layer name (ResNet)", value="conv5_block3_out")
    st.write("If your model uses a different backbone or layer name, change it here.")

# Show a load button to avoid auto download on page load
col1, col2 = st.columns([1, 1])
with col1:
    load_button = st.button("Load model from Drive")
with col2:
    st.write("")  # spacing

model = None
if load_button:
    if not file_id_input:
        st.error("Please paste the Drive file id in the sidebar first.")
    else:
        with st.spinner("Downloading and loading model (first time may take a minute)..."):
            model = load_model_from_drive(file_id_input)
            if model is not None:
                st.success("Model loaded successfully!")
            else:
                st.error("Model failed to load. Check Drive link, sharing permissions, and model format (.h5).")

# Allow uploading an image
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Uploaded image", use_column_width=True)
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    if model is None:
        st.warning("Model not loaded. Click 'Load model from Drive' in the sidebar and wait until it finishes.")
    else:
        # Predict + GradCAM
        with st.spinner("Running prediction..."):
            try:
                # Preprocess for prediction
                processed = preprocess_for_model(pil_img, target_size=(224, 224))

                preds = model.predict(processed)
                pred_idx = int(np.argmax(preds[0]))
                confidence = float(np.max(preds[0]))

                # Class names - edit as needed
                CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3"]
                # if length mismatch, fallback to numbers
                if pred_idx >= len(CLASS_NAMES):
                    predicted_name = f"class_{pred_idx}"
                else:
                    predicted_name = CLASS_NAMES[pred_idx]

                st.subheader("📌 Prediction Report")
                st.write(f"**Predicted class:** {predicted_name}")
                st.write(f"**Confidence:** {confidence:.4f}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # Grad-CAM
        with st.spinner("Generating Grad-CAM..."):
            try:
                preprocessed_for_gradcam = preprocess_for_model(pil_img, target_size=(224, 224))
                heatmap = make_gradcam_heatmap(preprocessed_for_gradcam, model, last_conv_layer_name=last_conv_layer_name)

                # convert PIL image to BGR for OpenCV overlay
                orig_bgr = cv2.cvtColor(np.array(pil_img.resize((224, 224))), cv2.COLOR_RGB2BGR)
                superimposed = overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.4)

                # Display side-by-side original + heatmap
                colA, colB = st.columns(2)
                with colA:
                    st.image(pil_img, caption="Original", use_column_width=True)
                with colB:
                    # convert back to RGB for display
                    disp = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
                    st.image(disp, caption="Grad-CAM Overlay", use_column_width=True)

                # Prepare downloadable PNG
                is_success, buffer = cv2.imencode(".png", superimposed)
                if is_success:
                    bytes_io = io.BytesIO(buffer.tobytes())
                    st.download_button(
                        label="⬇️ Download Grad-CAM (PNG)",
                        data=bytes_io,
                        file_name="gradcam.png",
                        mime="image/png"
                    )
                else:
                    st.warning("Could not prepare downloadable image.")

            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")
