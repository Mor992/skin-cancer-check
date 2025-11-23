# app.py
import streamlit as st
import requests
import tempfile
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import zipfile

st.set_page_config(page_title="Skin Lesion Classifier (auto model load)", layout="centered")

# ---------------- CONFIG: put your new file id here ----------------
FILE_ID = "1kJWpQQlF-2Rtwj2xRmVtbDw-83cyjD3q"   # <-- your Drive file id
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3"]  # change to your labels
LAST_CONV_LAYER = "conv5_block3_out"  # default for ResNet50

# ---------------- Helper: download (handles confirmation token) ----------------
def download_from_drive(file_id: str, chunk_size=1024*1024):
    base = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    resp = session.get(base, params={"id": file_id}, stream=True)

    # check for confirm token (large files)
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    if token:
        resp = session.get(base, params={"id": file_id, "confirm": token}, stream=True)

    # save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False)
    with tmp as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
    return tmp.name, resp.headers.get("Content-Type", "")

# ---------------- detect hdf5 signature ----------------
def is_hdf5(path):
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return head.startswith(b'\x89HDF\r\n\x1a\n')
    except Exception:
        return False

# ---------------- try to load model with fallbacks ----------------
@st.cache_resource
def load_model_auto(file_id: str):
    st.info("Downloading model from Google Drive...")
    model_path, content_type = download_from_drive(file_id)

    # Quick check: if server returned HTML (content-type text/html) -> fail early
    if content_type and "text/html" in content_type.lower():
        raise RuntimeError(f"Drive returned HTML (content-type: {content_type}). Check sharing settings (must be 'Anyone with link').")

    st.info("Checking file signature...")
    if is_hdf5(model_path):
        st.info("HDF5 signature found — loading .h5 model...")
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            # fallthrough to try SavedModel unzip
            st.warning(f".h5 load failed: {e} — will try alternative loaders.")
    else:
        st.warning("No HDF5 signature detected — trying alternative loading strategies.")

    # If it's a zip (maybe saved_model zipped), try to unzip and load
    try:
        # attempt to treat as zip
        tempdir = tempfile.mkdtemp()
        with zipfile.ZipFile(model_path, 'r') as z:
            z.extractall(tempdir)
        # if saved_model format present
        if os.path.exists(os.path.join(tempdir, "saved_model.pb")):
            st.info("Found SavedModel files inside the archive — loading SavedModel...")
            model = tf.keras.models.load_model(tempdir)
            return model
        # try to find a .h5 inside the zip extraction
        for root, _, files in os.walk(tempdir):
            for fname in files:
                if fname.endswith(".h5") or fname.endswith(".keras") or fname.endswith(".hdf5"):
                    candidate = os.path.join(root, fname)
                    st.info(f"Found {fname} inside archive — loading it...")
                    model = tf.keras.models.load_model(candidate)
                    return model
    except zipfile.BadZipFile:
        st.warning("Not a zip file.")
    except Exception as e:
        st.warning(f"Archive-based loading failed: {e}")

    # As last attempt: try tf.saved_model.load (non-keras)
    try:
        st.info("Attempting tf.saved_model.load on the downloaded path...")
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # final failure
        raise RuntimeError(f"Failed to load model automatically: {e}")

# ---------------- Grad-CAM utilities ----------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER):
    # assumes img_array is preprocessed and shaped (1,H,W,3)
    try:
        backbone = model.get_layer("resnet50")
    except Exception:
        raise RuntimeError("Model does not contain a 'resnet50' layer. Adjust LAST_CONV_LAYER or use a compatible model.")

    last_conv = backbone.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.input], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_outputs[0].numpy()
    pooled = pooled.numpy()

    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
    for i in range(pooled.shape[0]):
        heatmap += pooled[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.4):
    img = np.array(pil_img.convert("RGB"))
    hmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hmap = np.uint8(255 * hmap)
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(hmap, alpha, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

def preprocess_image_for_model(pil_img, target_size=(224,224)):
    img = pil_img.resize(target_size)
    arr = np.array(img).astype("float32")
    # normalize to model expectation: if values in 0..255, scale
    if arr.max() > 2.0:
        arr = arr / 255.0
    # some models expect resnet preprocessing; but most trained with 0..1 works.
    return np.expand_dims(arr, axis=0)

# ---------------- App UI ----------------
st.title("Automatic Skin Lesion Classifier (auto model load)")
st.write("App will automatically download your model from Google Drive and offer prediction + Grad‑CAM.")

# attempt to load model automatically
model = None
load_error = None
with st.spinner("Downloading and loading model — this may take a minute..."):
    try:
        model = load_model_auto = load_model_auto  # placeholder to satisfy linter
        model = load_model_auto(FILE_ID) if False else None  # not executed
    except Exception:
        pass

# The above attempt used a placeholder to keep cache semantics; we'll call load_model_auto properly below
if "model_loaded" not in st.session_state:
    try:
        st.session_state.model = load_model_auto(FILE_ID)
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model = None
        st.session_state.model_loaded = False
        st.error(f"Model load failed: {e}")
        st.stop()

model = st.session_state.model

st.success("Model is ready.")


uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded:
    try:
        pil = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Cannot read uploaded file: {e}")
        st.stop()

    st.image(pil, caption="Uploaded image", use_column_width=True)

    with st.spinner("Running prediction..."):
        try:
            x = preprocess_image_for_model(pil)
            preds = model.predict(x)
            idx = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0]))
            label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            st.subheader("Prediction")
            st.write(f"**Class:** {label}")
            st.write(f"**Confidence:** {conf:.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    with st.spinner("Generating Grad‑CAM..."):
        try:
            x_for_cam = preprocess_image_for_model(pil)
            heatmap = make_gradcam_heatmap(x_for_cam, model, LAST_CONV_LAYER)
            overlay = overlay_heatmap_on_image(pil, heatmap, alpha=0.45)
            st.subheader("Grad‑CAM")
            st.image(overlay, use_column_width=True)

            # download button
            is_ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if is_ok:
                st.download_button("⬇️ Download Grad‑CAM (PNG)", data=buf.tobytes(), file_name="gradcam.png", mime="image/png")
            else:
                st.warning("Could not prepare downloadable image.")
        except Exception as e:
            st.error(f"Grad‑CAM generation failed: {e}")
