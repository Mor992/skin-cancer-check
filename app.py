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
MODEL_DRIVE_ID = "1Rw-X-K2o75B70rT2Dwo-WFeA_RvXXWI3"   # Your file ID
MODEL_PATH = "resnet_skin_cancer.h5"

st.title("Skin Lesion Classifier with Grad-CAM++")


# ---------------------------------------------------
# GRAD-CAM++ IMPLEMENTATION
# ---------------------------------------------------
def get_last_conv_layer(model):
    # Recursively scan model for any Conv2D layer
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
        # Handle nested models (ResNet, EfficientNet etc.)
        if hasattr(layer, "layers"):
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    last_conv = sublayer.name
    return last_conv


def compute_gradcam_pp(model, img_array):
    # Find last conv layer reliably
    last_conv_layer = get_last_conv_layer(model)
    if last_conv_layer is None:
        return None

    conv_layer = model.get_layer(last_conv_layer)

    # Build Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    if grads is None:
        return None

    # Grad-CAM++ calculations
    grads_power2 = tf.square(grads)
    grads_power3 = grads_power2 * grads

    sum_conv = tf.reduce_sum(conv_output * grads_power3, axis=(1, 2), keepdims=True)
    alpha_denom = 2 * grads_power2 + sum_conv
    alpha_denom = tf.where(alpha_denom == 0, tf.ones_like(alpha_denom), alpha_denom)
    alphas = grads_power2 / alpha_denom

    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_output, axis=-1)[0]

    cam = tf.nn.relu(cam).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam



def overlay_heatmap(image, heatmap):
    image = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cm.jet(heatmap_resized)[:, :, :3]
    overlay = 0.5 * heatmap_color + 0.5 * (image / 255.0)
    return np.clip(overlay, 0, 1)


# ---------------------------------------------------
# DOWNLOAD + LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    # Download if missing
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model...")
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Try loading the model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("Model loaded successfully.")
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
uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    target_size = model.input_shape[1:3]
    img_resized = img.resize(target_size)
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    # Predict
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

    st.write("Full class probabilities:")
    for name, p in zip(class_names, pred):
        st.write(f"{name}: {p:.4f}")

    # Grad-CAM++
    st.subheader("Grad-CAM++ Visualization")
    heatmap = compute_gradcam_pp(model, arr)

    if heatmap is None:
        st.warning("Unable to compute Grad-CAM++ for this model.")
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
