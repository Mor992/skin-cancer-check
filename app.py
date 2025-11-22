import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple Grad-CAM", layout="centered")
st.title("Grad-CAM for ResNet")

# ----------------------------------------------------------
# Load model (auto-load, no buttons)
# ----------------------------------------------------------
MODEL_PATH = "final_resnet_model.keras"

st.info("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ----------------------------------------------------------
# Functions
# ----------------------------------------------------------

def preprocess(img, target=(224, 224)):
    img = img.convert("RGB").resize(target)
    arr = np.array(img).astype("float32")
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return preprocess_input(arr)

def gradcam(array, model, layer_name="conv5_block3_out"):
    x = tf.expand_dims(array, 0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv, preds = grad_model(x)
        top = tf.argmax(preds[0])
        target = preds[:, top]

    grads = tape.gradient(target, conv)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv = conv[0]
    heatmap = conv @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(top), float(preds[0][top])

def overlay_heatmap(img, heatmap):
    heatmap = Image.fromarray((heatmap * 255).astype("uint8")).resize(img.size)
    heatmap = np.array(heatmap)

    import matplotlib.cm as cm
    colored = cm.jet(heatmap / 255.0)[:, :, :3]
    colored = (colored * 255).astype("uint8")

    return Image.blend(img.convert("RGBA"), Image.fromarray(colored).convert("RGBA"), alpha=0.4)

# ----------------------------------------------------------
# Upload image
# ----------------------------------------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

classes_text = st.text_input("Class names (comma separated, optional):")
classes = [c.strip() for c in classes_text.split(",")] if classes_text else None

layer_name = st.text_input("Last conv layer name:", "conv5_block3_out")

if uploaded:
    img = Image.open(uploaded)
    arr = preprocess(img)

    try:
        heatmap, idx, score = gradcam(arr, model, layer_name)

        label = classes[idx] if classes and idx < len(classes) else str(idx)
        st.subheader(f"Prediction: {label} (score={score:.4f})")

        st.image(img, caption="Original Image", use_column_width=True)
        st.image(overlay_heatmap(img, heatmap), caption="Grad-CAM", use_column_width=True)

        fig, ax = plt.subplots()
        ax.imshow(heatmap)
        ax.axis("off")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")
