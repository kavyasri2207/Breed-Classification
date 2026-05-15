import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import pandas as pd

# ==============================
# LOAD MODELS
# ==============================
yolo_model = YOLO("yolov8s.pt")

MODEL_PATH = "breed_classifier_mobilenet (2).h5"

# Create folders
os.makedirs("flagged_for_learning", exist_ok=True)
os.makedirs("training_queue", exist_ok=True)

# Breed data
BREED_DATA = {
    "Bhadawari": {}, "Gir": {}, "Jaffarabadi": {},
    "Kankrej": {}, "Murrah": {}, "Nagpuri": {},
    "Ongole": {}, "Red_Sindhi": {}, "Sahiwal": {}, "Toda": {}
}

BREED_ORIGIN = {
    "Murrah": ["haryana", "punjab"],
    "Gir": ["gujarat"],
    "Sahiwal": ["punjab"],
    "Ongole": ["andhra"],
    "Kankrej": ["gujarat", "rajasthan"],
    "Nagpuri": ["maharashtra"],
}

CLASS_NAMES = sorted(BREED_DATA.keys())

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None

# ==============================
# YOLO DETECTION
# ==============================
def detect_animals(img):
    results = yolo_model(img, conf=0.35)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    img_area = img.size[0] * img.size[1]

    candidates = []

    for box, cls, score in zip(boxes, classes, scores):
        if int(cls) == 19:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            # 🚨 STRONG PRIORITY (area dominant)
            area_ratio = area / img_area

            # Penalize small detections heavily
            if area_ratio < 0.05:
                continue

            priority = (area_ratio * 0.7) + (score * 0.3)

            candidates.append((box, score, priority))

    # Sort by priority
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    final_boxes = [c[0] for c in candidates]
    final_scores = [c[1] for c in candidates]

    return final_boxes, final_scores

def enhance_display_image(img):
    # Resize for UI
    img = img.resize((900, 600))

    # Apply same enhancements as output
    img = ImageEnhance.Sharpness(img).enhance(1.6)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Brightness(img).enhance(1.05)

    return img
    
# ==============================
# DRAW BOXES
# ==============================
def draw_boxes(img, boxes, scores):
    img_np = np.array(img)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label = f"{score*100:.1f}%"
        cv2.putText(img_np, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Convert back to PIL for enhancement
    img_pil = Image.fromarray(img_np)

    # 🔥 ENHANCEMENTS
    img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.8)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.2)
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.05)

    return img_pil

# ==============================
# CLASSIFICATION
# ==============================
def classify(img, user_location):
    model = load_model()
    if model is None:
        return None, 0, None

    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)

    preds = model.predict(arr)[0]

    top_idx = np.argsort(preds)[-3:][::-1]
    top1, top2 = preds[top_idx[0]], preds[top_idx[1]]

    # Decision logic
    if top1 < 0.65:
        if (top1 - top2) < 0.12:
            label = "Possible Hybrid Breed"
        else:
            label = "Unknown"
    elif (top1 - top2) < 0.18:
        label = "Possible Hybrid Breed"
    else:
        label = CLASS_NAMES[top_idx[0]]

   
    confidence = float(top1)

    # Geo boost
    if label in BREED_ORIGIN:
        if any(loc in user_location.lower() for loc in BREED_ORIGIN[label]):
            confidence = min(confidence + 0.1, 0.99)

    return label, confidence, preds

# ==============================
# UI CONFIG
# ==============================
st.set_page_config(layout="wide")
st.markdown("""
<style>
div.stButton > button {
    border-radius:8px;
    font-weight:bold;
}

/* Submit = Blue */
div[data-testid="column"]:nth-of-type(1) button {
    background-color:#1e40af !important;
    color:white !important;
}

/* Delete = Red */
div[data-testid="column"]:nth-of-type(2) button {
    background-color:#dc3545 !important;
    color:white !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    app_mode = st.radio("Menu", ["Dashboard", "Analyzer", "Learning Lab"])
    user_location = st.selectbox(
    "Location",
    ["Andhra Pradesh","Gujarat","Punjab","Haryana","Rajasthan","Maharashtra","Other"]
    )
# ==============================
# DASHBOARD
# ==============================
if app_mode == "Dashboard":
    st.title("🐄 Bovine Intelligence System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📸 Smart Detection  
        Detect cows & buffaloes using YOLOv8  
        """)

    with col2:
        st.markdown("""
        ### 🧠 Breed Classification  
        Identify 10 Indian breeds using AI  
        """)

    with col3:
        st.markdown("""
        ### 🔄 Learning System  
        Improve model with user feedback  
        """)

    st.markdown("---")

    st.markdown("""
    ### 🚀 How it works
    1. Upload or capture image  
    2. Detect animals automatically  
    3. Classify breed using AI  
    4. Review uncertain cases in Learning Lab  
    """)

    st.markdown("---")

    st.success("✔ Supports multiple animals | ✔ Hybrid detection | ✔ Download reports")

# ==============================
# ANALYZER
# ==============================
elif app_mode == "Analyzer":

    st.title("🔍 Breed Analyzer")

    input_type = st.radio("Input", ["Upload", "Camera"], horizontal=True)

    file = st.file_uploader("Upload Image", type=None) \
        if input_type=="Upload" else st.camera_input("Capture")

    if file:
        try:
            img = Image.open(file).convert("RGB")
        except:
            st.error("Invalid image file")
            st.stop()

        display_img = enhance_display_image(img)

        # SIDE BY SIDE UI
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📥 Input Image")
            st.image(display_img, use_container_width=True)

        with col2:
            st.markdown("### 🧠 Detection Output")
            output_placeholder = st.empty()  # placeholder

        if st.button("Analyze"):

            with st.spinner("Analyzing..."):

                boxes, scores = detect_animals(img)

                if len(boxes) == 0:
                    st.error("🚫 No cows and buffaloes detected")
                else:
                    # DRAW OUTPUT
                    boxed = draw_boxes(img, boxes, scores)
                    boxed_display = enhance_display_image(boxed)

                    with col2:
                        output_placeholder.image(boxed_display, use_container_width=True)

                    # LIMIT GRID SIZE (better UX)
                    st.markdown("---")
                    st.markdown("### 🐄 Detected Animals")

                    cols = st.columns(min(len(boxes), 4))

                    results_list = []

                    for idx, (box, col) in enumerate(zip(boxes, cols)):
                        x1, y1, x2, y2 = map(int, box)
                        crop = img.crop((x1, y1, x2, y2)).resize((300, 300))

                        label, conf, preds = classify(crop, user_location)

                        results_list.append((idx+1, label, conf))

                        with col:
                            color = "green" if conf > 0.75 else "orange" if conf > 0.6 else "red"

                            st.markdown(
                                f"""
                                <div style="
                                    border:2px solid {color};
                                    border-radius:10px;
                                    padding:10px;
                                    text-align:center;
                                    background:#ffffff;">
                                """,
                                unsafe_allow_html=True
                            )

                            st.image(crop)
                            st.markdown(f"### {label}")
                            st.markdown(f"Confidence: **{conf*100:.1f}%**")

                            # ✅ CHART INSIDE CARD (clean UX)
                            if label not in ["Unknown", "Possible Hybrid Breed", "Ambiguous"]:
                                chart_data = {
                                    CLASS_NAMES[j]: float(preds[j])
                                    for j in range(len(CLASS_NAMES))
                                }
                                st.bar_chart(chart_data)

                            st.markdown("</div>", unsafe_allow_html=True)

                            # FLAG UNCERTAIN CASES
                            if label in ["Unknown", "Possible Hybrid Breed", "Ambiguous"]:
                                path = f"flagged_for_learning/{time.time()}.jpg"
                                crop.save(path)

                    # ======================
                    # 📥 REPORT DOWNLOAD
                    # ======================
                    report = [
                        {
                            "Animal": r[0],
                            "Prediction": r[1],
                            "Confidence": f"{r[2]*100:.2f}%"
                        }
                        for r in results_list
                    ]

                    df = pd.DataFrame(report)
                    csv = df.to_csv(index=False).encode("utf-8")

                    st.markdown("---")
                    st.download_button(
                        "📥 Download Report",
                        csv,
                        "report.csv",
                        "text/csv"
                    )

# ==============================
# LEARNING LAB
# ==============================
elif app_mode == "Learning Lab":

    st.title("🧪 Learning Lab")

    images = os.listdir("flagged_for_learning")

    if not images:
        st.info("No flagged images")
    else:
        selected = st.selectbox("Select Image", images)

        path = os.path.join("flagged_for_learning", selected)
        img = Image.open(path)

        st.image(img)

        label = st.selectbox("Correct Label", ["Unknown"]+CLASS_NAMES)

        c1,c2 = st.columns(2)

        with c1:
            if st.button("Submit"):
                save_dir = f"training_queue/{label}"
                os.makedirs(save_dir, exist_ok=True)
               
                img.save(f"{save_dir}/{selected}")
                st.info("Data saved. Model retraining required externally.")
                os.remove(path)

                st.success("Saved")
                st.rerun()
                

        with c2:
            if st.button("Delete"):
                os.remove(path)
                st.warning("Deleted")
                st.rerun()
