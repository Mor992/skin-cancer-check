import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import tempfile
import requests

st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")

# ---------------------------------------
# 1) LOAD MODEL FROM GOOGLE DRIVE LINK
# ---------------------------------------
@st.cache_resource
def load_model_from_drive():
    import requests
    import tempfile
    url = "https://drive.google.com/file/d/1uHgOzbvTY8hus4_ApzLlv7VO-Ye5uWpX/view?usp=drive_link"

    # Download file
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to download model! Check link permissions.")
        return None

    # Save into temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp:
        temp.write(response.content)
        temp_path = temp.name

    # Load model
    return tf.keras.models.load_model(temp_path)


model = load_model_from_drive()

st.title("🔥 Skin Lesion Classification with Grad‑CAM")
st.write("Upload an image and get prediction + visual explanation")

# ---------------------------------------
# 2) IMAGE UPLOAD
# ---------------------------------------
uploaded_file = st.file_uploader("Upload skin image", type=['jpg','jpeg','png'])

# CLASS NAMES (edit if needed)
CLASS_NAMES = ["benign", "malignant", "nevus", "other"]

# ---------------------------------------
# 3) GRADCAM FUNCTION
# ---------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    # Access the ResNet50 inside your model
    backbone = model.get_layer("resnet50")
    last_conv_layer = backbone.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [backbone.input],
        [last_conv_layer.output, backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap


def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    super_imposed = heatmap * 0.4 + img
    return np.uint8(super_imposed)


# ---------------------------------------
# 4) PROCESS IMAGE & PREDICT
# ---------------------------------------
if uploaded_file is not None:

    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prepare for model
    img_resized = cv2.resize(img, (224, 224))
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)

    st.subheader("📌 Prediction")
    st.write(f"**Class:** {CLASS_NAMES[pred_idx]}")
    st.write(f"**Confidence:** {preds[0][pred_idx]:.4f}")

    # Grad-CAM
    try:
        heatmap = make_gradcam_heatmap(img_array, model)
        img_with_heatmap = overlay_heatmap(img, heatmap)

        st.subheader("🔥 Grad-CAM Heatmap")
        st.image(img_with_heatmap, use_column_width=True)
    except Exception as e:
        st.error(f"GradCAM failed: {e}")

