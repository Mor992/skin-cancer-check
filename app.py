# app.py
import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Skin Cancer ResNet + Grad-CAM", layout="centered")

# ---------- Download model from Google Drive ----------
MODEL_PATH = "resnet_skin_cancer.h5"
DRIVE_FILE_ID = "1Rw-X-K2o75B70rT2Dwo-WFeA_RvXXWI3"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    st.info("Downloading model from Google Drive. This may take a minute...")
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded.")

# ---------- Load model ----------
@st.cache_resource
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

model = load_model(MODEL_PATH)

# ---------- Grad-CAM functions ----------
def get_gradcam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs],
                                      [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., np.newaxis], (img_array.shape[1], img_array.shape[2]))
    return cam.numpy(), int(pred_index)

def preprocess_uploaded(img, target_size=(224,224)):
    img = image.load_img(img, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr/255.0, axis=0)
    return img, arr

# ---------- UI ----------
st.title("Skin Cancer Detection (ResNet) + Grad-CAM")

uploaded_file = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
last_conv_layer_name = st.text_input("ResNet last conv layer name", value="conv5_block3_out",
                                    help="Change only if your model's last conv layer has a different name")

if uploaded_file:
    with st.spinner("Processing..."):
        pil_img, img_array = preprocess_uploaded(uploaded_file, target_size=(224,224))
        preds = model.predict(img_array)
        prob = np.max(preds[0])
        cls = np.argmax(preds[0])
        st.markdown(f"**Predicted class:** {cls}  \n**Confidence:** {prob:.3f}")

        cam, pred_index = get_gradcam(model, img_array, last_conv_layer_name)

        # Plot overlay
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(pil_img)
        ax.imshow(cam.squeeze(), cmap='jet', alpha=0.5)
        ax.axis('off')
        st.pyplot(fig)
