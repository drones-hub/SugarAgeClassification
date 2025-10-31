import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title
st.title("🌾 Sugarcane Age Prediction App")

# Load model (cache it so it doesn’t reload each time)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_model_noopt.keras")

")
    return model

model = load_model()

# Image upload section
uploaded_file = st.file_uploader("Upload a sugarcane field image (RGB)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # use your model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Predicting age..."):
        prediction = model.predict(img_array)
        predicted_age = round(float(prediction[0][0]), 2)  # adjust based on your output layer

    # Display result
    st.success(f"Predicted Sugarcane Age: **{predicted_age} months** 🌱")
