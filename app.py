import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import gdown
# ===========================
# 1. Load Model From Drive
# ===========================


ZIP_PATH = "my_model.keras"
MODEL_PATH = "my_model.keras"   # inside the zip

if not os.path.exists(MODEL_PATH):  # only download if model not extracted
     
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?export=download&id=1csR51feB60Uvzh3Qp4iMYuV9C0EkSMy_"
        gdown.download(url, MODEL_PATH, quiet=False)


    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")


model = tf.keras.models.load_model(MODEL_PATH)

# Your class names
CLASS_NAMES = ["Melanoma", "Nevus", "Seborrheic Keratosis"]

# ===========================
# 2. Helper Functions
# ===========================

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def generate_gradcam(model, img_array, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]

    grads = tape.gradient(pred_output, conv_output)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_output, guided_grads), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap.numpy()

def overlay_heatmap(original, heatmap):
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# ===========================
# 3. Simple Class Reports
# ===========================
CLASS_REPORTS = {
    "Melanoma": "Melanoma is a serious form of skin cancer. Early detection is important. "
                "If this is a real clinical case, seek professional evaluation.",
    "Nevus": "A nevus is a common mole. Most are benign, but unusual changes should be evaluated.",
    "Seborrheic Keratosis": "A benign skin growth often appearing in adults. Usually harmless."
}

# ===========================
# 4. Streamlit UI
# ===========================

st.title("Skin Lesion Classifier (ResNet50)")
st.write("Upload an image to classify it into skin lesion categories.")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])

show_gradcam = st.checkbox("Show Grad-CAM explanation (optional)")

if uploaded_file is not None:

    # Display uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", width=300)

    # Preprocess
    input_img = preprocess_image(img_rgb)

    # Prediction
    preds = model.predict(input_img)[0]
    pred_class = CLASS_NAMES[np.argmax(preds)]
    pred_prob = np.max(preds)

    # ===========================
    # Show Results
    # ===========================
    st.subheader("Prediction")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Probability:** {pred_prob:.4f}")

    st.subheader("Class Probabilities")
    for name, prob in zip(CLASS_NAMES, preds):
        st.write(f"{name}: {prob:.4f}")

    # ===========================
    # Optional Grad-CAM
    # ===========================
    if show_gradcam:
        st.subheader("Grad-CAM Visualization")

        heatmap = generate_gradcam(model, input_img)
        overlay = overlay_heatmap(img_rgb, heatmap)

        st.image(overlay, caption="Grad-CAM Heatmap", width=300)

    # ===========================
    # Medical-style Text Report
    # ===========================
    st.subheader("Information About Prediction")
    st.info(CLASS_REPORTS.get(pred_class, "No report available for this class."))
