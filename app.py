import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("Sugarcane Age Prediction App")

uploaded_file = st.file_uploader("Upload a sugarcane image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    model = tf.keras.models.load_model("final_model_noopet.keras")

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    st.success(f"Predicted Age: {prediction[0][0]:.2f} months")
else:
    st.info("Please upload an image to start prediction.")
