import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_model_noopt.keras")  # your trained model file
    return model

model = load_model()

# ----------------------------
# Define constants
# ----------------------------
AGE_CLASSES = ["2 month", "4 month", "6 month", "9 month", "11 month"]
PATCH_SIZE = 250  # for stitched image
INPUT_SIZE = (224, 224)

# ----------------------------
# Predict function
# ----------------------------
def predict_age(image):
    img_resized = cv2.resize(image, INPUT_SIZE)
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    preds = model.predict(img_resized)
    return AGE_CLASSES[np.argmax(preds)], np.max(preds)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌾 Sugarcane Age Prediction App")
st.write("Upload an RGB drone image to predict the crop age (2, 4, 6, 9, or 11 months).")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    img = np.array(image)

    st.write("### 🔍 Prediction Results:")

    # If stitched image (large), crop into patches
    if img.shape[0] > 1000 or img.shape[1] > 1000:
        st.write("Large image detected — cropping into patches...")
        patch_predictions = []

        for y in range(0, img.shape[0], PATCH_SIZE):
            for x in range(0, img.shape[1], PATCH_SIZE):
                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                if patch.shape[0] < 50 or patch.shape[1] < 50:
                    continue
                age, conf = predict_age(patch)
                patch_predictions.append(age)

        # Majority vote
        if patch_predictions:
            final_age = max(set(patch_predictions), key=patch_predictions.count)
            st.success(f"Predicted Age: **{final_age}** (based on {len(patch_predictions)} patches)")
        else:
            st.warning("No valid patches found in image.")

    else:
        # Single image prediction
        predicted_age, confidence = predict_age(img)
        st.success(f"Predicted Age: **{predicted_age}** ({confidence*100:.2f}% confidence)")
