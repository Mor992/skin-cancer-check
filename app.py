import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import tempfile
import cv2
import io
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")

# -----------------------------------------------------
# 1) Download model from Google Drive (AUTO)
# -----------------------------------------------------
FILE_ID = "1kJWpQQlF-2Rtwj2xRmVtbDw-83cyjD3q"   # final_resnet_model.keras

@st.cache_resource
def load_model():
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    session = requests.Session()

    st.info("Downloading model from Google Drive...")
    response = session.get(url, stream=True)

    # HANDLE GOOGLE DRIVE CONFIRMATION TOKEN
    def get_confirm_token(r):
        for key, value in r.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm={token}"
        response = session.get(url, stream=True)

    # SAVE TEMP FILE
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                temp.write(chunk)
        model_path = temp.name

    st.success("Model downloaded. Loading...")

    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None


model = load_model()

# -----------------------------------------------------
# 2) Grad‑CAM
# -----------------------------------------------------
def make_gradcam_heatmap(img_array, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    heatmap = np.zeros(conv_outputs.shape[0:2])

    for i in range(pooled_grads.shape[0]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    return heatmap


def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap_color, 0.4, img, 0.6, 0)


# -----------------------------------------------------
# 3) UI
# -----------------------------------------------------
st.title("🔥 Skin Lesion Classifier + Grad‑CAM (Final Working App)")
st.write("Model: **final_resnet_model.keras** (auto‑loaded)")

uploaded = st.file_uploader("Upload skin image", type=["jpg", "jpeg", "png"])

if uploaded and model is not None:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, caption="Uploaded Image", use_column_width=True)

    img = pil.resize((224, 224))
    arr = np.array(img)[..., :3]
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    # Prediction
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))

    st.subheader("📌 Prediction")
    st.write(f"**Class:** {idx}")
    st.write(f"**Confidence:** {conf:.4f}")

    # Grad‑CAM
    heatmap = make_gradcam_heatmap(arr, model)
    orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cam = overlay_heatmap(orig, heatmap)

    cam_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    st.subheader("🔥 Grad‑CAM Visualization")
    st.image(cam_rgb, use_column_width=True)

    # Download button
    ok, buf = cv2.imencode(".png", cam)
    if ok:
        st.download_button(
            label="⬇️ Download Grad‑CAM",
            data=buf.tobytes(),
            file_name="gradcam.png",
            mime="image/png"
        )
