import streamlit as st
st.set_page_config(layout="wide", page_title="Bovine Intelligence System")
import cv2
import numpy as np
import os
import time
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import pandas as pd
from fpdf import FPDF
import io

# Flag to indicate if breed classification is available
CLASSIFIER_AVAILABLE = os.path.exists("breed_classifier_mobilenet (2).h5")

# Breed detailed information for Encyclopedia
BREED_INFO = {
    "Bhadawari": {"Origin": "Uttar Pradesh & Madhya Pradesh", "Type": "Buffalo", "Yield": "800 - 1000 kg", "Fat": "Up to 13%", "Desc": "Known for high fat content in milk. Medium-sized body, copperish color."},
    "Gir": {"Origin": "Gujarat", "Type": "Cattle", "Yield": "1500 - 2500 kg", "Fat": "4.5 - 5%", "Desc": "Renowned dairy breed, highly disease resistant. Prominent forehead and long, pendulous ears."},
    "Jaffarabadi": {"Origin": "Gujarat", "Type": "Buffalo", "Yield": "1800 - 2700 kg", "Fat": "8 - 8.5%", "Desc": "One of the heaviest buffalo breeds. Horns are heavy and broad, drooping downwards."},
    "Kankrej": {"Origin": "Gujarat & Rajasthan", "Type": "Cattle", "Yield": "1500 - 1800 kg", "Fat": "4.8%", "Desc": "Dual-purpose breed (milk and draught). Large, strong, with massive lyre-shaped horns."},
    "Murrah": {"Origin": "Haryana & Punjab", "Type": "Buffalo", "Yield": "1500 - 2500 kg", "Fat": "7%", "Desc": "The most famous dairy buffalo in the world. Jet black color with tightly curled horns."},
    "Nagpuri": {"Origin": "Maharashtra", "Type": "Buffalo", "Yield": "700 - 1200 kg", "Fat": "7 - 8%", "Desc": "Long, flat, sword-shaped horns. Used for both milk and draught power."},
    "Ongole": {"Origin": "Andhra Pradesh", "Type": "Cattle", "Yield": "500 - 1000 kg", "Fat": "4 - 5%", "Desc": "Large, muscular, and disease-resistant. Known globally and used to develop the Brahman breed."},
    "Red_Sindhi": {"Origin": "Sindh (Pakistan) & India", "Type": "Cattle", "Yield": "1500 - 2500 kg", "Fat": "4.5 - 5%", "Desc": "Excellent dairy breed, highly adaptable to different climates. Deep red color."},
    "Sahiwal": {"Origin": "Punjab", "Type": "Cattle", "Yield": "2000 - 3000 kg", "Fat": "4.5%", "Desc": "Considered the best indigenous dairy breed. Reddish-brown color, highly tick-resistant."},
    "Toda": {"Origin": "Tamil Nadu (Nilgiri Hills)", "Type": "Buffalo", "Yield": "500 - 800 kg", "Fat": "8%", "Desc": "Distinctive semi-wild breed kept by the Toda tribe. Thick hair coat and wide, bowed horns."}
}

def create_pdf_report(results_list, original_image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Bovine Intelligence System - Report", ln=1, align='C')
    pdf.ln(5)
    
    temp_img_path = f"temp_report_{time.time()}.jpg"
    original_image.save(temp_img_path)
    pdf.image(temp_img_path, x=30, w=150)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt=f"Total Animals Detected: {len(results_list)}", ln=1)
    pdf.ln(2)
    
    pdf.set_font("Arial", size=11)
    for idx, label, conf in results_list:
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(200, 8, txt=f"Animal #{idx}: {label} ({conf*100:.1f}%)", ln=1)
        if label in BREED_INFO:
            info = BREED_INFO[label]
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 6, txt=f"Type: {info['Type']} | Origin: {info['Origin']}\nDesc: {info['Desc']}")
        pdf.ln(4)
        
    pdf_path = f"report_{time.time()}.pdf"
    pdf.output(pdf_path)
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        
    return pdf_bytes

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
# LOAD MODEL (Conditional - TensorFlow optional)
# ==============================
@st.cache_resource
def load_model():
    if CLASSIFIER_AVAILABLE:
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except ImportError:
            return None
        except Exception:
            return None
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
        return "Model Not Available", 0.0, None

    try:
        import tensorflow as tf
        # Prepare image
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        # MobileNet preprocessing: normalize to [-1, 1]
        arr = arr / 127.5 - 1.0

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
    except:
        return "Model Not Available", 0.0, None

# ==============================
# UI CONFIG
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
div.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white !important;
    border: none;
    box-shadow: 0 4px 6px -1px rgba(30, 64, 175, 0.2);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(30, 64, 175, 0.3);
}
div[data-testid="column"]:nth-of-type(2) button {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
}
h1, h2, h3 {
    background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    border-radius: 15px; padding: 20px; border: 1px solid rgba(59, 130, 246, 0.2);
    text-align: center; transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
}
.custom-card {
    border: none; border-top: 5px solid; border-radius: 15px; padding: 15px;
    text-align: center; background: #ffffff; box-shadow: 0 10px 20px rgba(0,0,0,0.08);
    transition: transform 0.2s; margin-bottom: 20px;
}
.custom-card:hover { transform: translateY(-5px); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    app_mode = st.radio("Menu", ["Dashboard", "Analyzer", "Breed Encyclopedia", "Learning Lab"])
    user_location = st.selectbox(
        "Location",
        ["Andhra Pradesh","Gujarat","Punjab","Haryana","Rajasthan","Maharashtra","Other"]
    )

# ==============================
# DASHBOARD
# ==============================
if app_mode == "Dashboard":
    st.title("🐄 Bovine Intelligence System")
    st.markdown("<p style='font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem;'>Advanced AI-powered detection and breed classification for Indian cattle and buffaloes.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h1 style="margin:0;">📸</h1><h3 style="-webkit-text-fill-color: initial;color:#1f2937;">Detection</h3><p style="color:#6b7280;">YOLOv8</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card"><h1 style="margin:0;">🧬</h1><h3 style="-webkit-text-fill-color: initial;color:#1f2937;">10 Breeds</h3><p style="color:#6b7280;">MobileNet AI</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card"><h1 style="margin:0;">🔄</h1><h3 style="-webkit-text-fill-color: initial;color:#1f2937;">Learning</h3><p style="color:#6b7280;">User Feedback</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("""
        ### 🚀 Getting Started
        1. Navigate to the **Analyzer** tab.
        2. **Upload** an image or **Capture** a photo.
        3. The system detects and classifies the animals.
        4. Download a **PDF Report** of the results!
        5. Read more in the **Breed Encyclopedia**.
        """)
        st.success("✨ New: PDF Export, Encyclopedia & Premium UI added!")
        
    with c2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e40af, #8b5cf6); border-radius: 20px; padding: 40px; text-align: center; color: white;">
            <h1 style="-webkit-text-fill-color: white; font-size: 60px; margin:0;">🐮</h1>
            <h2 style="-webkit-text-fill-color: white; margin-top:10px;">AI Precision</h2>
        </div>
        """, unsafe_allow_html=True)

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
        except Exception as e:
            st.error(f"Invalid image file: {e}")
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

                    for idx, box in enumerate(boxes):
                        col = cols[idx % 4] # Wrap around columns if >4 animals detected
                        x1, y1, x2, y2 = map(int, box)
                        crop = img.crop((x1, y1, x2, y2)).resize((300, 300))

                        label, conf, preds = classify(crop, user_location)

                        results_list.append((idx+1, label, conf))

                        with col:
                            color = "green" if conf > 0.75 else "orange" if conf > 0.6 else "red"

                            st.markdown(
                                f"""
                                <div class="custom-card" style="border-top-color: {color};">
                                """,
                                unsafe_allow_html=True
                            )

                            st.image(crop)
                            st.markdown(f"### {label}")
                            st.markdown(f"Confidence: **{conf*100:.1f}%**")

                            # ✅ CHART INSIDE CARD (clean UX)
                            if label not in ["Unknown", "Possible Hybrid Breed", "Ambiguous", None]:
                                chart_data = {
                                    CLASS_NAMES[j]: float(preds[j])
                                    for j in range(len(CLASS_NAMES))
                                }
                                st.bar_chart(chart_data)
                            elif label is None:
                                st.error("Model file not found!")

                            st.markdown("</div>", unsafe_allow_html=True)

                            # FLAG UNCERTAIN CASES
                            if label in ["Unknown", "Possible Hybrid Breed", "Ambiguous"]:
                                path = f"flagged_for_learning/{time.time()}.jpg"
                                crop.save(path)

                    # ======================
                    # 📥 REPORT DOWNLOAD
                    # ======================
                    report = [{"Animal": r[0], "Prediction": r[1], "Confidence": f"{r[2]*100:.2f}%"} for r in results_list]
                    df = pd.DataFrame(report)
                    csv = df.to_csv(index=False).encode("utf-8")

                    st.markdown("---")
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button("📊 Download CSV Report", csv, "report.csv", "text/csv", use_container_width=True)
                    
                    with col_dl2:
                        pdf_bytes = create_pdf_report(results_list, boxed)
                        st.download_button("📄 Download PDF Report", pdf_bytes, "report.pdf", "application/pdf", use_container_width=True)

# ==============================
# BREED ENCYCLOPEDIA
# ==============================
elif app_mode == "Breed Encyclopedia":
    st.title("📖 Breed Encyclopedia")
    st.markdown("Explore information about the 10 Indian cattle and buffalo breeds supported by our AI.")
    selected_breed = st.selectbox("Select a Breed to explore:", CLASS_NAMES)
    
    if selected_breed in BREED_INFO:
        info = BREED_INFO[selected_breed]
        st.markdown(f"## {selected_breed}")
        st.markdown(
            f"""
            <div style="background-color:rgba(30, 64, 175, 0.1); padding:20px; border-radius:10px; border-left: 5px solid #1e40af; margin-bottom: 20px;">
                <h3 style="color:#1e40af; margin-top:0;">{info['Type']}</h3>
                <p><strong>📍 Origin:</strong> {info['Origin']}</p>
                <p><strong>🥛 Avg. Milk Yield:</strong> {info['Yield']}</p>
                <p><strong>🧈 Fat Content:</strong> {info['Fat']}</p>
                <p><strong>📝 Description:</strong> {info['Desc']}</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.info("💡 **Tip:** Use the 'Analyzer' tab to upload an image and let AI identify it!")

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
                img.close()
                os.remove(path)

                st.success("Saved")
                st.rerun()
                

        with c2:
            if st.button("Delete"):
                img.close()
                os.remove(path)
                st.warning("Deleted")
                st.rerun()
