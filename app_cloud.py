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

# ============================== 
# BREED INFORMATION
# ==============================
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

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_yolo():
    try:
        return YOLO("yolov8s.pt")
    except Exception as e:
        return None

# ==============================
# DETECTION FUNCTION
# ==============================
def detect_animals(img, yolo_model):
    if yolo_model is None:
        st.error("YOLO model not available")
        return [], []
        
    try:
        results = yolo_model(img, conf=0.35)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        return boxes, scores
    except Exception as e:
        st.error(f"Detection error: {e}")
        return [], []

# ==============================
# UI STYLING
# ==============================
st.markdown("""
<style>
h1 { color: #3b82f6; }
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# MAIN APP
# ==============================
st.title("Bovine Intelligence System")
st.write("Detect and classify cattle and buffalo breeds using AI")

# Load YOLO
yolo_model = load_yolo()

# Sidebar navigation
app_mode = st.sidebar.selectbox("Choose Feature", ["Dashboard", "Analyzer", "Encyclopedia"])

if app_mode == "Dashboard":
    st.header("Welcome to Bovine Intelligence System")
    st.write("""
    This application helps you:
    - Detect cattle and buffalo in images
    - Classify breed types
    - Get detailed breed information
    - Generate reports
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Breeds", len(BREED_INFO))
    with col2:
        st.metric("Detection Model", "YOLOv8")
    with col3:
        st.metric("Status", "Online")

elif app_mode == "Analyzer":
    st.header("Animal Detection & Analysis")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Animals"):
            with st.spinner("Analyzing..."):
                boxes, scores = detect_animals(image, yolo_model)
                
                if len(boxes) > 0:
                    st.success(f"Detected {len(boxes)} animal(s)")
                    for i, (box, score) in enumerate(zip(boxes, scores)):
                        st.write(f"Animal {i+1}: Confidence {score*100:.1f}%")
                else:
                    st.warning("No animals detected")

elif app_mode == "Encyclopedia":
    st.header("Breed Encyclopedia")
    
    breed = st.selectbox("Select Breed", list(BREED_INFO.keys()))
    
    if breed:
        info = BREED_INFO[breed]
        st.write(f"**Type:** {info['Type']}")
        st.write(f"**Origin:** {info['Origin']}")
        st.write(f"**Milk Yield:** {info['Yield']}")
        st.write(f"**Fat Content:** {info['Fat']}")
        st.write(f"**Description:** {info['Desc']}")

st.sidebar.markdown("---")
st.sidebar.write("**Note:** For full breed classification, use the local deployment")
