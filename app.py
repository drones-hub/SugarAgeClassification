import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="🌿 Sugarcane Age Classifier",
    page_icon="🍃",
    layout="centered"
)

st.title('🌿 Sugarcane Age Classification')
st.write("Upload a sugarcane field or plant image to classify its growth stage.")

# --- Model Loading ---
@st.cache_resource
def load_our_model():
    """Loads the trained sugarcane age prediction model."""
    try:
        model = load_model("final_model_noopt.keras")  # ✅ using your working model
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_our_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    """Resize and normalize image for model input."""
    target_size = (240, 240)  # ✅ model expects 240x240
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Class Labels (Match your model training order) ---
CLASS_NAMES = ['2 month', '4 month', '6 month', '9 month', '11 month']

# --- File Upload ---
uploaded_file = st.file_uploader("📸 Choose a sugarcane image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='📷 Uploaded Image', use_column_width=True)
        st.write("🔍 Classifying...")

        # Preprocess and Predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(prediction) * 100

        # --- Display Results ---
        st.success(f"✅ **Predicted Age:** {predicted_class}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")

        # Optional: show all class probabilities
        st.write("### Prediction Probabilities:")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

elif model is None:
    st.warning("⚠️ Model could not be loaded. Please verify 'final_model_noopt.keras' exists in your app folder.")
else:
    st.info("📂 Please upload an image to start prediction.")
