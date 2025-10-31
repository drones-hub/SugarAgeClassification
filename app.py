import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

st.set_page_config(page_title="Sugarcane Age Prediction", layout="centered")
st.title("🌿 Sugarcane Age Prediction App")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model_noopt.keras")

model = load_model()

# Define age classes
AGE_CLASSES = ["2 month", "4 month", "6 month", "9 month", "11 month"]
PATCH_SIZE = 240  # based on your model input shape

def predict_age(image):
    """Preprocess image and predict sugarcane age."""
    image = Image.fromarray(image).resize((PATCH_SIZE, PATCH_SIZE))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    confidence = np.max(preds)
    return AGE_CLASSES[predicted_index], confidence

# File uploader
uploaded_file = st.file_uploader("📸 Upload a sugarcane image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convert file buffer safely into an image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)

        # Show uploaded image
        st.image(img_array, caption="Uploaded Image", use_column_width=True)  # ✅ fixed here

        # If stitched image, crop into patches
        if img_array.shape[0] > 1000 or img_array.shape[1] > 1000:
            st.info("🧩 Large image detected — analyzing patches...")
            patch_predictions = []

            for y in range(0, img_array.shape[0], PATCH_SIZE):
                for x in range(0, img_array.shape[1], PATCH_SIZE):
                    patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    if patch.shape[0] < PATCH_SIZE // 2 or patch.shape[1] < PATCH_SIZE // 2:
                        continue
                    age, _ = predict_age(patch)
                    patch_predictions.append(age)

            if patch_predictions:
                final_age = max(set(patch_predictions), key=patch_predictions.count)
                st.success(f"Predicted Age: **{final_age}** (based on {len(patch_predictions)} patches)")
            else:
                st.warning("⚠️ No valid patches found in this image.")
        else:
            # Single prediction
            predicted_age, conf = predict_age(img_array)
            st.success(f"✅ Predicted Age: **{predicted_age}** ({conf*100:.2f}% confidence)")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

else:
    st.info("📂 Please upload a sugarcane image to start prediction.")
