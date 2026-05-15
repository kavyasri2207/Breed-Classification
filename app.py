import streamlit as st
st.set_page_config(layout="wide", page_title="Bovine Intelligence System")

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    HAS_TF = True
except ImportError:
    HAS_TF = False

import cv2
import numpy as np
import os
import time
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import csv
import io
import base64
from fpdf import FPDF

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
try:
    yolo_model = YOLO("yolov8s.pt")
except:
    yolo_model = None

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
    if yolo_model is None:
        return [], []
    # Lowered confidence from 0.35 to 0.15 to catch tiny/low-res images like the Bing URL
    results = yolo_model(img, conf=0.15)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    img_area = img.size[0] * img.size[1]

    candidates = []

    for box, cls, score in zip(boxes, classes, scores):
        # Allow Cow(19), Elephant(20), Bear(21) which YOLO confuses Buffalo for.
        # Removed Sheep(18) and Horse(17) so it doesn't classify literal sheep!
        if int(cls) in [19, 20, 21]:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            # 🚨 STRONG PRIORITY (area dominant)
            area_ratio = area / img_area

            # Lowered penalty so tiny images pass through
            if area_ratio < 0.01:
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
    # Apply enhancements
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

    img_pil = Image.fromarray(img_np)
    img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.8)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.2)
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.05)
    return img_pil

# ==============================
# CLASSIFICATION
# ==============================
def classify(img, user_location):
    if not HAS_TF:
        st.error("TensorFlow is not installed. Breed classification disabled.")
        return "Unknown (TF Missing)", 0.0, [0.0]*len(CLASS_NAMES)

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
# UI CONFIG & THEME
# ==============================
with st.sidebar:
    app_mode = st.radio("Menu", ["Dashboard", "Analyzer", "Breed Encyclopedia", "Learning Lab", "About Project"])
    user_location = st.selectbox(
        "Location",
        ["Andhra Pradesh","Gujarat","Punjab","Haryana","Rajasthan","Maharashtra","Other"]
    )

# Keeping the premium Dark Mode UI/UX explicitly
bg_style = """
background-color: #0d1117;
background-image: radial-gradient(at 0% 0%, rgba(14, 53, 114, 0.4) 0px, transparent 50%),
                  radial-gradient(at 100% 0%, rgba(105, 75, 23, 0.4) 0px, transparent 50%);
"""
card_bg = "rgba(22, 27, 34, 0.7)"
border_color = "rgba(48, 54, 61, 0.5)"
text_primary = "#f0f6fc"
text_secondary = "#8b949e"
tab_active_bg = "rgba(59, 46, 21, 0.9)"
tab_active_text = "#e2b340"
tab_hover_text = "#c9d1d9"

css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');

/* NUCLEAR TEXT COLOR OVERRIDE for Streamlit Native Elements */
* {{
    color: {text_primary} !important;
    font-family: 'Inter', sans-serif;
}}

/* EXCEPTIONS TO NUCLEAR FIX */
.stButton button, .stButton button * {{
    color: #ffffff !important;
}}
.photo-banner, .photo-banner *, .photo-banner h2 {{
    color: #ffffff !important;
    text-shadow: 0 4px 8px rgba(0,0,0,0.8) !important;
}}
div[data-baseweb="tab"][aria-selected="true"], div[data-baseweb="tab"][aria-selected="true"] * {{
    color: {tab_active_text} !important;
}}
p, span, label, li, small {{
    color: {text_secondary} !important;
}}
h1, h2, h3, h4 {{
    color: {text_primary} !important;
}}

/* PROTECT IMAGE FULLSCREEN CONTROLS */
[data-testid="StyledFullScreenButton"], [data-testid="StyledFullScreenButton"] *,
button[title="View fullscreen"], button[title="View fullscreen"] *,
button[title="Close fullscreen"], button[title="Close fullscreen"] * {{
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
}}

.stApp {{
    {bg_style}
    background-attachment: fixed;
}}
.block-container {{
    padding-top: 3rem !important;
}}

/* FIX SIDEBAR BACKGROUND */
[data-testid="stSidebar"] {{
    background-color: {card_bg} !important;
}}
[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}

/* FIX DROPDOWN AND SELECTBOX BACKGROUNDS */
div[role="listbox"], div[data-baseweb="popover"] {{
    background-color: {card_bg} !important;
}}
div[data-baseweb="select"] > div {{
    background-color: {card_bg} !important;
}}

/* FIX TEXT INPUTS (URL BOX) */
div[data-baseweb="input"] {{
    background-color: transparent !important;
}}
div[data-baseweb="input"] > div {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
}}
div[data-baseweb="input"] input {{
    color: {text_primary} !important;
    background-color: transparent !important;
}}
div[data-baseweb="input"] input::placeholder {{
    color: {text_secondary} !important;
    opacity: 0.6 !important;
}}

/* Tabs */
div[data-testid="stTabs"] {{
    background: {card_bg};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 12px;
    padding: 8px;
    border: 1px solid {border_color};
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}}
div[data-baseweb="tab-list"] {{
    gap: 10px;
    justify-content: center;
}}
div[data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {text_secondary} !important;
    border-radius: 8px !important;
    padding: 12px 30px !important;
    border: none !important;
    font-weight: 600;
}}
div[data-baseweb="tab"]:hover {{
    color: {tab_hover_text} !important;
}}
div[data-baseweb="tab"][aria-selected="true"] {{
    background: {tab_active_bg} !important;
    color: {tab_active_text} !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display: none; }}

/* Uploader Dropzone */
[data-testid="stFileUploader"] > section {{
    background: {card_bg} !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 2px dashed {border_color} !important;
    border-radius: 16px !important;
    padding: 40px !important;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}}
[data-testid="stFileUploader"] > section:hover {{
    border-color: #e2b340 !important;
}}
[data-testid="stFileUploader"] button {{
    background: linear-gradient(135deg, #a16207 0%, #ca8a04 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
}}
[data-testid="stFileUploader"] button * {{
    color: #ffffff !important;
}}

/* Primary Button */
div.stButton > button {{
    width: 100%;
    background: linear-gradient(135deg, #a16207 0%, #ca8a04 100%) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    padding: 20px !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    border: none !important;
    margin-top: 10px;
    margin-bottom: 20px;
    transition: all 0.2s ease;
    box-shadow: 0 10px 20px rgba(161, 98, 7, 0.3) !important;
}}
div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 15px 25px rgba(161, 98, 7, 0.4) !important;
}}

/* Metric Cards */
.metric-card {{
    background: {card_bg};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 15px; padding: 20px; border: 1px solid {border_color};
    text-align: center; transition: all 0.3s ease;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}}
.metric-card:hover {{
    border-color: #e2b340;
    transform: translateY(-3px);
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ==============================
# DASHBOARD
# ==============================
if app_mode == "Dashboard":
    st.markdown(f"<h1 style='color: {text_primary}; font-size: 3rem; font-weight: 800; margin-bottom: 0px;'>🐄 Bovine Intelligence System</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_secondary}; font-size: 1.1rem; margin-top: 5px; margin-bottom: 2rem;'>Advanced AI-powered detection and breed classification for Indian cattle and buffaloes.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="metric-card"><h1 style="margin:0;">📸</h1><h3 style="color: {text_primary};">Detection</h3><p style="color: {text_secondary};">YOLOv8</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="metric-card"><h1 style="margin:0;">🧬</h1><h3 style="color: {text_primary};">10 Breeds</h3><p style="color: {text_secondary};">MobileNet AI</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown(f'<div class="metric-card"><h1 style="margin:0;">🔄</h1><h3 style="color: {text_primary};">Learning</h3><p style="color: {text_secondary};">User Feedback</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown(f"""
        <h3 style="color: {text_primary};">🚀 Getting Started</h3>
        <p style="color: {text_secondary};">1. Navigate to the <b>Analyzer</b> tab.</p>
        <p style="color: {text_secondary};">2. <b>Upload</b> an image or <b>Capture</b> a photo.</p>
        <p style="color: {text_secondary};">3. The system detects and classifies the animals.</p>
        <p style="color: {text_secondary};">4. Download a <b>PDF Report</b> of the results!</p>
        <p style="color: {text_secondary};">5. Read more in the <b>Breed Encyclopedia</b>.</p>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="photo-banner" style="
            background-image: url('https://images.unsplash.com/photo-1545468800-85cc9bc6ecf7?q=80&w=1000&auto=format&fit=crop');
            background-size: cover;
            background-position: center;
            border-radius: 20px; 
            padding: 80px 40px; 
            text-align: center;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
            border: 2px solid {border_color};
        ">
            <div style="position: absolute; top:0; left:0; right:0; bottom:0; background: rgba(0,0,0,0.5);"></div>
            <h2 style="margin-top:10px; color: white !important; position: relative; z-index: 1; text-shadow: 0 4px 8px rgba(0,0,0,0.8); font-size: 2rem;">Discover The Breed</h2>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# ANALYZER
# ==============================
elif app_mode == "Analyzer":
    st.markdown(f"<h1 style='color: {text_primary}; font-size: 3.5rem; font-weight: 800; margin-bottom: 0px;'>Predict Breed</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_secondary}; font-size: 1.1rem; margin-top: 5px; margin-bottom: 2rem;'>Upload an image, use your camera, or paste a URL to identify the breed.</p>", unsafe_allow_html=True)

    img_data = None
    
    # Modern Tabbed Interface for Inputs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Camera", "🔗 Image URL"])
    
    with tab1:
        file = st.file_uploader(
            "📤 Click to upload or drag & drop (JPG, PNG, WebP — max 10MB)", 
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible"
        )
        if file:
            img_data = file
            
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        enable_camera = st.checkbox("📸 Enable Camera Hardware")
        if enable_camera:
            camera_file = st.camera_input("Capture a photo of the animal")
            if camera_file:
                img_data = camera_file
        else:
            st.info("Check the box above to grant camera access.")
            
    with tab3:
        url = st.text_input("Paste a direct link to an image:")
        if url:
            import requests
            import io
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                img_data = io.BytesIO(response.content)
            except:
                st.error("❌ Failed to load image from URL. Please ensure it is a direct link to an image file.")

    if img_data:
        try:
            img = Image.open(img_data).convert("RGB")
        except Exception as e:
            st.error(f"Invalid image format.")
            st.stop()

        display_img = enhance_display_image(img)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📥 Input Image")
            st.image(display_img, use_container_width=True)

        with col2:
            st.markdown("### 🧠 Detection Output")
            output_placeholder = st.empty()

        if st.button("🔍 Classify Breed"):
            # Laser Scanner Animation
            with col2:
                scan_placeholder = st.empty()
                buffered = io.BytesIO()
                display_img.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                
                scan_placeholder.markdown(f"""
                <div style="position: relative; border-radius: 16px; overflow: hidden; border: 2px solid #e2b340;">
                    <img src="data:image/jpeg;base64,{img_b64}" style="width: 100%; display: block;">
                    <div style="
                        position: absolute;
                        top: 0; left: 0; right: 0; height: 4px;
                        background: #e2b340;
                        box-shadow: 0 0 20px 5px rgba(226, 179, 64, 0.8);
                        animation: scan 1.2s infinite alternate ease-in-out;
                    "></div>
                </div>
                <style>
                @keyframes scan {{
                    0% {{ top: 0%; opacity: 1; }}
                    100% {{ top: 98%; opacity: 1; }}
                }}
                </style>
                """, unsafe_allow_html=True)
                
            # Artificial delay so they see the cool scanner!
            time.sleep(1.5)
            
            with st.spinner("Finalizing Classification..."):
                boxes, scores = detect_animals(img)
                
                # Clear the scanner before showing real results
                scan_placeholder.empty()

                if len(boxes) == 0:
                    st.error("🚫 No cows and buffaloes detected")
                else:
                    boxed = draw_boxes(img, boxes, scores)
                    boxed_display = enhance_display_image(boxed)

                    with col2:
                        output_placeholder.image(boxed_display, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### 🐄 Detected Animals")

                    cols = st.columns(min(len(boxes), 4))
                    results_list = []

                    for idx, box in enumerate(boxes):
                        col = cols[idx % 4]
                        x1, y1, x2, y2 = map(int, box)
                        crop = img.crop((x1, y1, x2, y2)).resize((300, 300))

                        label, conf, preds = classify(crop, user_location)
                        results_list.append((idx+1, label, conf))

                        with col:
                            color = "green" if conf > 0.75 else "orange" if conf > 0.6 else "red"
                            st.markdown(
                                f'<div class="custom-card" style="border-top-color: {color};">',
                                unsafe_allow_html=True
                            )
                            st.image(crop)
                            st.markdown(f"### {label}")
                            st.markdown(f"Confidence: **{conf*100:.1f}%**")

                            if label not in ["Unknown", "Possible Hybrid Breed", "Ambiguous", None, "Unknown (TF Missing)"]:
                                st.markdown("##### Top Predictions:")
                                top_idx = np.argsort(preds)[-3:][::-1]
                                for j in range(3):
                                    st.write(f"- **{CLASS_NAMES[top_idx[j]]}**: {preds[top_idx[j]]*100:.1f}%")
                            elif label is None:
                                st.error("Model file not found!")

                            st.markdown("</div>", unsafe_allow_html=True)

                            if label in ["Unknown", "Possible Hybrid Breed", "Ambiguous"]:
                                path = f"flagged_for_learning/{time.time()}.jpg"
                                crop.save(path)

                    # ======================
                    # 📥 REPORT DOWNLOAD
                    # ======================
                    report = [{"Animal": r[0], "Prediction": r[1], "Confidence": f"{r[2]*100:.2f}%"} for r in results_list]
                    
                    # Manual CSV generation without Pandas
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=["Animal", "Prediction", "Confidence"])
                    writer.writeheader()
                    writer.writerows(report)
                    csv_data = output.getvalue().encode('utf-8')

                    st.markdown("---")
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button("📊 Download CSV Report", csv_data, "report.csv", "text/csv", use_container_width=True)
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

# ==============================
# ABOUT PROJECT
# ==============================
elif app_mode == "About Project":
    st.markdown(f"<h1 style='color: {text_primary}; font-size: 3rem; font-weight: 800; margin-bottom: 0px;'>ℹ️ About the Project</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_secondary}; font-size: 1.1rem; margin-top: 5px; margin-bottom: 2rem;'>An AI-powered cattle breed classification system for Indian indigenous breeds.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: left; min-height: 250px;">
            <h3 style="color: {text_primary}; margin-top: 0;">🧠 The Models</h3>
            <p style="color: {text_secondary};">A custom two-stage deep learning pipeline:</p>
            <ul style="color: {text_secondary};">
                <li><b>Detection:</b> YOLOv8 isolates the animal from background noise.</li>
                <li><b>Classification:</b> TensorFlow MobileNet analyzes features to determine the exact breed.</li>
                <li><b>Fallback:</b> Continuous feedback loop via the Learning Lab.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="text-align: left; min-height: 250px;">
            <h3 style="color: {text_primary}; margin-top: 0;">⚙️ Tech Stack</h3>
            <ul style="color: {text_secondary};">
                <li><b>AI/ML:</b> TensorFlow, Keras, Ultralytics YOLOv8</li>
                <li><b>Computer Vision:</b> OpenCV, PIL</li>
                <li><b>Frontend & UI:</b> Streamlit, Custom CSS (Glassmorphism)</li>
                <li><b>Reporting:</b> FPDF for automated PDF generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: left; min-height: 250px;">
            <h3 style="color: {text_primary}; margin-top: 0;">📊 The Dataset</h3>
            <p style="color: {text_secondary};">Focused on 10 highly distinct indigenous Indian breeds (8 cattle + 2 buffalo breeds).</p>
            <ul style="color: {text_secondary};">
                <li>Images dynamically resized to 224×224 pixels</li>
                <li>Data normalization applied for MobileNet</li>
                <li>Geographical Confidence Boosting algorithm applied</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="text-align: left; min-height: 250px;">
            <h3 style="color: {text_primary}; margin-top: 0;">🌾 For Farmers</h3>
            <p style="color: {text_secondary};">Designed for real-world agricultural use.</p>
            <ul style="color: {text_secondary};">
                <li>Camera capture for direct field use</li>
                <li>PDF Reporting for veterinary tracking</li>
                <li>Breed encyclopedia with milk yield data</li>
                <li>"Learning Lab" pipeline for misclassified animals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div class="metric-card" style="text-align: left; max-width: 800px; margin: 0 auto;">
        <h3 style="color: {text_primary}; margin-top: 0;">👤 Creator</h3>
        <p style="color: {text_secondary};">Built by <b>Kavya Sri</b>.</p>
        <p style="color: {text_secondary};"><a href="https://github.com/kavyasri2207" target="_blank" style="color: #e2b340; text-decoration: none; font-weight: bold;">🔗 View GitHub Profile</a></p>
        <hr style="border: 1px solid {border_color}; margin: 15px 0;">
        <ul style="color: {text_secondary}; margin-bottom: 0;">
            <li><b>Backend & Inference:</b> Pure Python with TensorFlow & OpenCV</li>
            <li><b>Frontend UI:</b> Streamlit with Custom CSS injection</li>
            <li><b>Continuous Learning:</b> Custom data-collection pipeline for iterative model improvement</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
