import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("🌾 Sugarcane Age Prediction App")

# -------------------------------
# Upload Section
# -------------------------------
uploaded_file = st.file_uploader("📸 Upload a sugarcane image", type=["jpg", "jpeg", "png"])

# -------------------------------
# Model Loading (safe)
# -------------------------------
MODEL_PATH = "final_model_noopt.keras"

# Check if model exists before loading
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Please ensure 'final_model_noopt.keras' is uploaded to the same directory as app.py.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)

    # -------------------------------
    # Prediction Section
    # -------------------------------
    if uploaded_file is not None:
        try:
            # Open and display image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess image
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Predict
            prediction = model.predict(img_array)
            predicted_age = float(prediction[0][0])

            st.success(f"🌱 Predicted Sugarcane Age: **{predicted_age:.2f} months**")

        except Exception as e:
            st.error(f"⚠️ Error processing image: {e}")

    else:
        st.info("📤 Please upload an image to start prediction.")
