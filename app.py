import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(
    page_title="🌿 Sugarcane Age Classifier",
    page_icon="🍃",
    layout="centered"
)

st.title("🌿 Sugarcane Age Classification")
st.write("Upload an image of a sugarcane plant to classify its growth stage.")

# -----------------------------
# Model Loading
# -----------------------------
@st.cache_resource
def load_model_once():
    """Load the trained Keras model once."""
    try:
        model = tf.keras.models.load_model("final_model_noopt.keras")  # Update path if needed
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model_once()

# -----------------------------
# Image Preprocessing Function
# -----------------------------
def preprocess_image(image):
    """Resize and normalize the image for model prediction."""
    try:
        img = image.convert("RGB")  # Ensure image is in RGB mode
        img = img.resize((240, 240))  # Match your model’s input size
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"⚠️ Error preprocessing image: {e}")
        return None

# -----------------------------
# Class Labels
# -----------------------------
CLASS_NAMES = ['2_month', '4_month', '6_month', '9_month', '11_month']

# -----------------------------
# File Upload & Prediction
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload a sugarcane image", type=["jpg", "jpeg", "png"])

if model is None:
    st.warning("⚠️ Model could not be loaded. Please verify model path and format.")
elif uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("🔍 Classifying...")

        # Preprocess and predict
        processed = preprocess_image(image)
        if processed is not None:
            prediction = model.predict(processed)
            predicted_index = int(np.argmax(prediction))
            predicted_label = CLASS_NAMES[predicted_index]
            confidence = float(np.max(prediction) * 100)

            # Display prediction
            st.success(f"✅ Predicted Age: **{predicted_label.replace('_', ' ')}** ({confidence:.2f}% confidence)")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
else:
    st.info("📂 Please upload a sugarcane image to start prediction.")
