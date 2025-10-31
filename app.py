import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Title
st.title("🌾 Sugarcane Age Prediction App")
st.write("Upload a sugarcane field image (RGB) to predict the crop age using the trained model.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/sugarcane_age_model.h5")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = img.resize((224, 224))   # Adjust if your model uses different input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict
    prediction = model.predict(img_array)
    predicted_age = prediction[0][0]  # If regression output
    st.success(f"🧮 Predicted Sugarcane Age: **{predicted_age:.2f} months**")
