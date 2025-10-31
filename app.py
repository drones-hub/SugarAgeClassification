import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="Sugarcane Age Prediction", page_icon="🌾")

st.title("🌾 Sugarcane Age Prediction App")
st.markdown("Upload an RGB image of sugarcane to predict crop age in months.")

# -------------------------------
# Load the model safely
# -------------------------------
MODEL_PATH = "final_model_noopt.keras"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Loading model... ⏳"):
            model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.warning("⚠️ Model file not found! Upload `final_model_noopt.keras` in the same directory as `app.py`.")

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("📸 Upload a sugarcane image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Proceed only if model loaded
        if model is not None:
            with st.spinner("Predicting... 🧠"):
                # Resize & preprocess
                img = image.resize((224, 224))
                img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

                # Predict
                prediction = model.predict(img_array)
                predicted_age = float(prediction[0][0])

            st.success(f"🌱 Predicted Sugarcane Age: **{predicted_age:.2f} months**")
        else:
            st.error("⚠️ Model not loaded. Please check your model file.")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
else:
    st.info("📤 Please upload an image to start prediction.")
