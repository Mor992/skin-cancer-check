import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# ---------------------------
# LOAD MODEL (LOCAL OR DRIVE)
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "https://drive.google.com/file/d/1PFrSDFTI7SI_f8JrxZ8hUcZb5jVyAnkD/view?usp=sharing"   # or path to Google Drive mounted file
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ---------------------------
# GRAD‑CAM FUNCTION
# ---------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        pred_idx = tf.argmax(predictions[0])
        loss = predictions[:, pred_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# ---------------------------
# PREPROCESS FUNCTION
# ---------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Skin Lesion Classifier + Grad‑CAM Heatmap")
st.subheader("Upload an image and get a prediction report")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if uploaded_file:
    # display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # preprocess
    arr = preprocess_image(img)

    # prediction
    preds = model.predict(arr)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])

    classes = ["Class 0", "Class 1", "Class 2", "Class 3"]  # ← replace with real labels

    st.subheader("Prediction Report")
    st.write(f"**Predicted Class:** {classes[class_idx]}")
    st.write(f"**Confidence:** {confidence:.4f}")

    # GRAD‑CAM
    heatmap = make_gradcam_heatmap(arr, model)

    # apply heatmap to image
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

    st.subheader("Grad‑CAM Heatmap")
    st.image(superimposed, use_column_width=True)
