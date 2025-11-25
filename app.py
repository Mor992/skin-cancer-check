import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import gdown
import os
from matplotlib import cm

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
MODEL_DRIVE_ID = "1XlZArIYbtkG3_NRyP2hsRBViY0C5T67f"
MODEL_PATH = "model.weights.h5"

st.title("Skin Lesion Classifier with Grad-CAM++")

# ---------------------------------------------------
# GRAD-CAM++ FUNCTIONS
# ---------------------------------------------------
def compute_gradcam_pp(model, img_array):
    conv_layers = [l.name for l in model.layers if "conv" in l.name]
    if not conv_layers:
        return None
    last_conv_layer = conv_layers[-1]

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img_array)
        pred_index = tf.argmax(prediction[0])
        loss = prediction[:, pred_index]

    grads = tape.gradient(loss, conv_output)
    grads_power_2 = tf.square(grads)
    grads_power_3 = grads_power_2 * grads

    sum_conv = tf.reduce_sum(conv_output * grads_power_3, axis=(1, 2), keepdims=True)
    alpha_denom = 2 * grads_power_2 + sum_conv
    alpha_denom = tf.where(alpha_denom != 0, alpha_denom, tf.ones_like(alpha_denom))

    alphas = grads_power_2 / alpha_denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(1, 2))

    cam = tf.reduce_sum(weights[:, None, None, :] * conv_output, axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay_heatmap(image, heatmap):
    image = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cm.jet(heatmap_resized)[:, :, :3]
    overlay = heatmap_color * 0.5 + image / 255.0
    return np.clip(overlay, 0, 1)


# ---------------------------------------------------
# DOWNLOAD + LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


model = load_model()
if model is None:
    st.stop()

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------
uploaded = st.file_uploader("Upload skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image")

    target_size = model.input_shape[1:3]
    img_resized = img.resize(target_size)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    # Prediction
    class_names = [
        "Melanoma → Cancer (malignant)",
        "Melanocytic Nevus → Usually benign",
        "Basal Cell Carcinoma → Cancer",
        "Actinic Keratosis → Pre-cancerous"
    ]

    pred = model.predict(arr)[0]
    idx = int(np.argmax(pred))
    confidence = float(pred[idx])

    st.subheader("Prediction Result")
    st.write(f"Top class: **{class_names[idx]}** ({confidence:.4f})")

    st.write("Full probabilities:")
    for name, p in zip(class_names, pred):
        st.write(f"{name}: {p:.4f}")

    st.subheader("Grad-CAM++")

    heatmap = compute_gradcam_pp(model, arr)
    if heatmap is None:
        st.warning("Grad-CAM++ could not be generated.")
    else:
        overlay = overlay_heatmap(img_resized, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Original")
            st.image(img)

        with col2:
            st.write("Grad-CAM++")
            st.image(overlay)

else:
    st.info("Upload an image to start classification.")

