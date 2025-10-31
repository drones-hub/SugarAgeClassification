# streamlit_app.py
import os
import logging
import warnings
from pathlib import Path
from io import BytesIO, StringIO
from PIL import Image, ImageDraw
import numpy as np
import tempfile

# Suppress noisy logs/warnings before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide INFO/WARNING from TF
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import streamlit as st
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown  # download shared file from Google Drive
from collections import Counter

# Streamlit config (unchanged)
st.set_page_config(
    page_title="Web Application for Sugarcane Age Detection using Drone Imagery",
    layout="wide"
)

# -------- Configuration (unchanged) --------
DRIVE_FILE_ID_DEFAULT = "10JYTIb9CWNhGbhnBNEA1Yj8SVVqx5BjE"
DEFAULT_MODEL_FILENAME = "/tmp/model.keras"
VALID_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
USE_VGG_PREPROCESS = False
TOP_K_DEFAULT = 3

# Default mapping
DEFAULT_CLASS_MAP = {
    0: "11_month",
    1: "2_month",
    2: "4_month",
    3: "6_month",
    4: "9_month"
}

# SVERI Logo URL
LOGO_URL = "https://coe.sveri.ac.in/wp-content/themes/SVERICoE/images/sverilogo.png"

# -------- Utility Functions (unchanged logic) --------
def get_model_input_size(model):
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if not shape:
        return (240, 240, 3)
    if len(shape) == 4:
        _, h, w, c = shape
        h = int(h) if (h is not None) else 240
        w = int(w) if (w is not None) else 240
        c = int(c) if (c is not None) else 3
        return (h, w, c)
    return (240, 240, 3)

def infer_num_classes_from_model(model):
    try:
        out_shape = model.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        if isinstance(out_shape, tuple) and len(out_shape) >= 2:
            n = out_shape[-1]
            if isinstance(n, int):
                return n
    except Exception:
        pass
    try:
        for layer in reversed(model.layers):
            if hasattr(layer, "units"):
                n = getattr(layer, "units")
                if isinstance(n, int) and n > 1:
                    return n
    except Exception:
        pass
    return None

def build_default_class_map(model, prefix="class_"):
    n = infer_num_classes_from_model(model)
    if n is None:
        return {}
    else:
        return {i: f"{prefix}{i}" for i in range(n)}

def preprocess_image_for_model_bytes(img_bytes, model, use_vgg=USE_VGG_PREPROCESS):
    expected_h, expected_w, expected_c = get_model_input_size(model)
    pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    pil_resized = pil.resize((expected_w, expected_h), Image.BILINEAR)
    x = np.array(pil_resized).astype("float32")
    if use_vgg:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        x = preprocess_input(x)
    else:
        x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x, pil_resized, (expected_h, expected_w, expected_c)

def predict_from_bytes(model, img_bytes, class_map=None, top_k=3, use_vgg=USE_VGG_PREPROCESS):
    if class_map is None:
        class_map = build_default_class_map(model)
    x, pil_img, used_size = preprocess_image_for_model_bytes(img_bytes, model, use_vgg=use_vgg)
    preds = model.predict(x, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = preds[0] if (hasattr(preds, "ndim") and preds.ndim == 2 and preds.shape[0] == 1) else preds
    probs = softmax(preds) if preds.sum() > 1.0001 or preds.min() < 0 else preds
    top_idx = probs.argsort()[-top_k:][::-1]
    results = [(int(idx), class_map.get(int(idx), f"class_{idx}"), float(probs[idx])) for idx in top_idx]
    return results, pil_img

def download_from_gdrive(file_id: str, dest_path: str, force=False):
    dest = Path(dest_path)
    if dest.exists() and not force:
        return str(dest)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest), quiet=False)
    return str(dest)

def load_model_preferred(path, convert_h5_to_keras=True, compile=False):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if p.suffix.lower() == ".keras":
        return load_model(str(p), compile=compile)
    elif p.suffix.lower() == ".h5" and convert_h5_to_keras:
        model = load_model(str(p), compile=compile)
        keras_path = str(p.with_suffix(".keras"))
        model.save(keras_path)
        return load_model(keras_path, compile=compile)
    else:
        return load_model(str(p), compile=compile)

@st.cache_resource(show_spinner=False)
def get_model_from_drive(drive_file_id=DRIVE_FILE_ID_DEFAULT, local_path=DEFAULT_MODEL_FILENAME, force=False):
    dest = download_from_gdrive(drive_file_id, local_path, force=force)
    model = load_model_preferred(dest, compile=False)
    return model, dest

# -------- Streamlit UI (header unchanged) --------
col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_URL, width=130)

with col2:
    st.markdown("## Web Application for Sugarcane Age Detection using Drone Imagery")
    st.markdown("**Developed by:** SVERI's College of Engineering, Pandharpur  ")
    st.markdown("**Research funding support from:** Rajiv Gandhi Science and Technology Commission, Government of Maharashtra")

st.markdown("---")

# About model (unchanged)
st.markdown(
    """
**About the Model (brief):**

This application uses a **MobileNetV2 backbone** fine-tuned on drone imagery of sugarcane fields.
It classifies sugarcane crop age into stages such as *2, 4, 6, 9,* and *11 months*.
The final layer is a dense classification head using Softmax activation.
The model was trained using annotated drone datasets collected across multiple farms.
"""
)

# Sidebar (unchanged)
with st.sidebar:
    st.header("Model / Prediction Settings")
    drive_id = st.text_input("Google Drive File ID", value=DRIVE_FILE_ID_DEFAULT)
    model_dest = st.text_input("Local Model Path", value=DEFAULT_MODEL_FILENAME)
    force_dl = st.checkbox("Force re-download model", value=False)
    top_k = st.number_input("Top K predictions", min_value=1, max_value=10, value=TOP_K_DEFAULT)
    use_vgg = st.checkbox("Use VGG preprocessing (/255.0 off)", value=USE_VGG_PREPROCESS)

# Load model (unchanged)
with st.spinner("Downloading and loading model..."):
    try:
        model, model_path = get_model_from_drive(drive_file_id=drive_id, local_path=model_dest, force=force_dl)
        st.success(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Apply class map automatically (unchanged)
model_classes = infer_num_classes_from_model(model)
if model_classes == len(DEFAULT_CLASS_MAP):
    class_map = DEFAULT_CLASS_MAP
    st.info("Using default sugarcane age mapping.")
else:
    class_map = build_default_class_map(model)
    st.warning("Default mapping size mismatch; using generic labels.")

# ----------------- NEW: single stitched-image uploader only -----------------
st.header("Upload a single stitched farm image (JPEG/PNG)")
stitched_file = st.file_uploader("Upload stitched image (one file only)", accept_multiple_files=False, type=["jpg", "jpeg", "png"])

if stitched_file is not None:
    try:
        stitched_image = Image.open(stitched_file).convert("RGB")
    except Exception as e:
        st.error(f"Failed to open uploaded image: {e}")
        stitched_image = None

    if stitched_image is not None:
        st.image(stitched_image, caption=f"Uploaded stitched image: {stitched_file.name}", use_column_width=True)
        st.write("---")
        st.write("### Tiling stitched image into 160x160 crops and classifying tiles...")

        # Tiling logic (preserve only full tiles, as before)
        crop_size = 160
        width, height = stitched_image.size
        cropped_images = []
        crop_boxes = []
        for y in range(0, height, crop_size):
            for x in range(0, width, crop_size):
                if x + crop_size <= width and y + crop_size <= height:
                    box = (x, y, x + crop_size, y + crop_size)
                    crop = stitched_image.crop(box)
                    cropped_images.append(crop)
                    crop_boxes.append(box)

        if not cropped_images:
            st.warning("The stitched image is smaller than 160x160 and could not be tiled.")
        else:
            # Build batch array for prediction
            batch_list = []
            inp_h, inp_w, inp_c = get_model_input_size(model)
            for crop in cropped_images:
                crop_resized = crop.resize((inp_w, inp_h), Image.BILINEAR)
                arr2 = np.array(crop_resized).astype("float32")
                if use_vgg:
                    from tensorflow.keras.applications.vgg16 import preprocess_input
                    arr2 = preprocess_input(arr2)
                else:
                    arr2 = arr2 / 255.0
                batch_list.append(arr2)
            batch_array = np.stack(batch_list, axis=0)

            # Predict
            preds = model.predict(batch_array, verbose=0)
            preds = preds[0] if (hasattr(preds, "ndim") and preds.ndim == 3 and preds.shape[0] == 1) else preds
            try:
                if preds.sum(axis=1).max() > 1.0001 or preds.min() < 0:
                    probs = softmax(preds, axis=1)
                else:
                    probs = preds
            except Exception:
                probs = preds

            predicted_indices = np.argmax(probs, axis=1)
            predicted_labels = [class_map.get(int(idx), f"class_{idx}") for idx in predicted_indices]

            # Count and percentage
            counts = Counter(predicted_labels)
            total_tiles = len(cropped_images)

            st.subheader("✅ Overall Prediction Summary")
            col1, col2 = st.columns(2)
            most_common_label, most_common_count = counts.most_common(1)[0]
            with col1:
                st.metric("Final Predicted Age (Majority Vote)", most_common_label)
            with col2:
                st.metric("Number of Tiles Analyzed", total_tiles)

            st.write("#### Prediction Breakdown (tile counts and percentage of field):")
            for lbl, cnt in counts.items():
                pct = (cnt / total_tiles) * 100
                st.write(f"- **{lbl}:** {cnt} tiles — **{pct:.2f}%** of field")

            st.write("---")

            # ----------------- NEW: overlay map -----------------
            # create RGBA overlay
            overlay = Image.new("RGBA", stitched_image.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)

            # choose distinct colors for each class (in order of class_map keys)
            # palette will be reused deterministically
            palette = [
                (31,119,180,140),   # blue-ish
                (255,127,14,140),   # orange
                (44,160,44,140),    # green
                (214,39,40,140),    # red
                (148,103,189,140),  # purple
                (140,86,75,140),    # brown
                (227,119,194,140),  # pink
                (127,127,127,140),  # gray
            ]
            # Map labels to colors
            unique_labels = list(dict.fromkeys(predicted_labels))  # preserve order
            label_to_color = {}
            for i, lbl in enumerate(sorted(list(set(class_map.values())))):  # stable ordering across runs
                color = palette[i % len(palette)]
                label_to_color[lbl] = color

            # Draw rectangles for each tile with color based on predicted label
            for box, lbl in zip(crop_boxes, predicted_labels):
                color = label_to_color.get(lbl, (0,0,0,120))
                draw.rectangle(box, fill=color, outline=None)

            # Composite overlay onto stitched image
            stitched_rgba = stitched_image.convert("RGBA")
            composited = Image.alpha_composite(stitched_rgba, overlay)

            st.subheader("Spatial Overlay Map (tiles colored by predicted class)")
            st.image(composited, caption="Overlay: semi-transparent tile predictions", use_column_width=True)

            # Legend: display color swatches + percentages
            st.write("#### Legend and Percentages")
            legend_cols = st.columns(len(label_to_color))
            for i, (lbl, color) in enumerate(label_to_color.items()):
                with legend_cols[i]:
                    sw = Image.new("RGBA", (50, 30), color)
                    st.image(sw, width=60)
                    cnt = counts.get(lbl, 0)
                    pct = (cnt/total_tiles)*100 if total_tiles>0 else 0.0
                    st.markdown(f"**{lbl}**  \n{cnt} tiles  \n{pct:.2f}%")

            st.write("---")

            # Individual tile grid (same as earlier but smaller thumbnails)
            st.subheader("Individual Tile Analysis")
            num_columns = 4
            cols = st.columns(num_columns)
            for i, (crop, prob_row) in enumerate(zip(cropped_images, probs)):
                col = cols[i % num_columns]
                pred_idx = int(np.argmax(prob_row))
                pred_label = class_map.get(pred_idx, f"class_{pred_idx}")
                confidence = float(np.max(prob_row))
                with col:
                    display_width = min(200, max(64, crop.width // 2))
                    st.image(crop, caption=f"Tile #{i+1}", width=display_width)
                    st.success(f"Prediction: {pred_label} ({confidence:.3f})")

            # CSV download with spatial coordinates and prediction
            results_all = {}
            rows = []
            for i, (box, prob_row) in enumerate(zip(crop_boxes, probs), start=1):
                x0,y0,x1,y1 = box
                pred_idx = int(np.argmax(prob_row))
                pred_label = class_map.get(pred_idx, f"class_{pred_idx}")
                prob_val = float(np.max(prob_row))
                fname = f"{stitched_file.name}_tile_{i}"
                results_all[fname] = [(pred_idx, pred_label, prob_val)]
                rows.append({
                    "tile_id": i,
                    "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1,
                    "predicted_label": pred_label,
                    "probability": prob_val
                })

            if st.button("Download Results (CSV)"):
                import pandas as pd
                df = pd.DataFrame(rows)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "stitched_predictions.csv", "text/csv")

else:
    st.info("Please upload a single stitched image (JPEG/PNG) to begin classification.")

# Footer (unchanged)
st.markdown("---")
st.markdown(
    """
**Project PI / Contact:**  
Dr. Prashant Maruti Pawar  
SVERI's College of Engineering, Pandharpur  
For collaboration or data access, please contact the institute.
"""
)
