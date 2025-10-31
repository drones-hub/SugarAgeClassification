# streamlit_app.py
import os
import logging
import warnings
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import streamlit as st
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Web Application for Sugarcane Age Detection using Drone Imagery",
    layout="wide"
)

# --- Constants ---
LOCAL_MODEL_PATH = "final_model_noopt.keras"   # ✅ Your local model file
VALID_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Class map (ensure order matches your model output)
CLASS_MAP = {
    0: "2_month",
    1: "4_month",
    2: "6_month",
    3: "9_month",
    4: "11_month"
}

LOGO_URL = "https://coe.sveri.ac.in/wp-content/themes/SVERICoE/images/sverilogo.png"

# --- Utility Functions ---
def get_model_input_size(model):
    """Get input size from model."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if not shape:
        return (240, 240, 3)
    if len(shape) == 4:
        _, h, w, c = shape
        return (int(h or 240), int(w or 240), int(c or 3))
    return (240, 240, 3)

def preprocess_image_bytes(img_bytes, model):
    """Preprocess single image bytes for model input."""
    h, w, _ = get_model_input_size(model)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_resized = img.resize((w, h))
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img_resized

@st.cache_resource(show_spinner=False)
def load_local_model(path=LOCAL_MODEL_PATH):
    """Load the local Keras model."""
    try:
        model = load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from '{path}': {e}")
        return None

# --- Header ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_URL, width=130)

with col2:
    st.markdown("## Web Application for Sugarcane Age Detection using Drone Imagery")
    st.markdown("**Developed by:** SVERI's College of Engineering, Pandharpur")
    st.markdown("**Supported by:** Rajiv Gandhi Science and Technology Commission, Govt. of Maharashtra")

st.markdown("---")

# --- Model Loading ---
with st.spinner("🔄 Loading model..."):
    model = load_local_model()
    if model is None:
        st.stop()
st.success(f"✅ Model loaded successfully: {LOCAL_MODEL_PATH}")

# --- Image Upload ---
st.header("📸 Upload a Stitched Farm Image (JPEG/PNG)")
stitched_file = st.file_uploader("Upload stitched image (one file only)", type=["jpg", "jpeg", "png"])

if stitched_file is not None:
    try:
        stitched_image = Image.open(stitched_file).convert("RGB")
        st.image(stitched_image, caption=f"Uploaded image: {stitched_file.name}", use_column_width=True)
    except Exception as e:
        st.error(f"⚠️ Error opening image: {e}")
        st.stop()

    st.write("---")
    st.write("### 🧩 Splitting image into 240×240 tiles and classifying...")

    crop_size = 240
    width, height = stitched_image.size
    tiles, boxes = [], []

    # Crop image into 240x240 patches
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            if x + crop_size <= width and y + crop_size <= height:
                box = (x, y, x + crop_size, y + crop_size)
                crop = stitched_image.crop(box)
                tiles.append(crop)
                boxes.append(box)

    if not tiles:
        st.warning("The image is smaller than 240×240. Please upload a larger image.")
        st.stop()

    # Prepare batch for prediction
    arrs = []
    h, w, _ = get_model_input_size(model)
    for t in tiles:
        arr = np.array(t.resize((w, h))).astype("float32") / 255.0
        arrs.append(arr)
    batch = np.stack(arrs, axis=0)

    # Predict
    preds = model.predict(batch, verbose=0)
    probs = softmax(preds, axis=1)
    pred_idx = np.argmax(probs, axis=1)
    labels = [CLASS_MAP.get(int(i), f"class_{i}") for i in pred_idx]

    # Summary stats
    counts = Counter(labels)
    total = len(tiles)
    common_label, common_count = counts.most_common(1)[0]

    st.subheader("✅ Field-Level Prediction Summary")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Dominant Age", common_label)
    col2.metric("Total Tiles", total)

    st.write("#### Breakdown:")
    for lbl, cnt in counts.items():
        pct = (cnt / total) * 100
        st.write(f"- **{lbl}:** {cnt} tiles — {pct:.2f}%")

    st.write("---")

    # --- Overlay Visualization ---
    st.subheader("🗺️ Overlay Visualization (Color-coded Tiles)")
    overlay = Image.new("RGBA", stitched_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colors = {
        "2_month": (255, 0, 0, 120),     # red
        "4_month": (255, 165, 0, 120),   # orange
        "6_month": (0, 128, 0, 120),     # green
        "9_month": (0, 0, 255, 120),     # blue
        "11_month": (128, 0, 128, 120)   # purple
    }

    for box, lbl in zip(boxes, labels):
        color = colors.get(lbl, (128, 128, 128, 120))
        draw.rectangle(box, fill=color)

    composite = Image.alpha_composite(stitched_image.convert("RGBA"), overlay)
    st.image(composite, caption="Predicted field overlay", use_column_width=True)

    # Legend
    st.write("#### Legend:")
    cols = st.columns(len(colors))
    for i, (lbl, color) in enumerate(colors.items()):
        with cols[i]:
            swatch = Image.new("RGBA", (40, 30), color)
            st.image(swatch, width=50)
            st.markdown(f"**{lbl}**")

    # Download CSV
    st.write("---")
    import pandas as pd
    rows = []
    for i, (box, lbl, prob) in enumerate(zip(boxes, labels, np.max(probs, axis=1))):
        rows.append({
            "tile_id": i + 1,
            "x_min": box[0], "y_min": box[1],
            "x_max": box[2], "y_max": box[3],
            "predicted_label": lbl,
            "probability": prob
        })
    df = pd.DataFrame(rows)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Tile Predictions (CSV)", csv, "sugarcane_predictions.csv", "text/csv")

else:
    st.info("📂 Please upload a stitched image to begin classification.")

# --- Footer ---
st.markdown("---")
st.markdown("""
**Project PI / Contact:**  
Dr. Prashant Maruti Pawar  
SVERI's College of Engineering, Pandharpur  
*Supported by Rajiv Gandhi Science and Technology Commission, Govt. of Maharashtra*
""")
