
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from utils import preprocess_image, make_prediction, generate_pdf_report
import gdown
import os
import streamlit as st
import streamlit as st

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Dynamic CSS injection based on dark_mode
def apply_custom_theme(dark):
    if dark:
        bg_color = "#0f0f0f"
        text_color = "#f5f5f5"
        card_color = "#1e1e1e"
        accent_color = "#bb86fc"
    else:
        bg_color = "#f5f5f5"
        text_color = "#000000"
        card_color = "#ffffff"
        accent_color = "#007acc"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
            transition: background-color 0.4s ease-in-out;
        }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {accent_color} !important;
        }}

        .css-18e3th9 {{
            background-color: {card_color} !important;
            color: {text_color} !important;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .stButton>button {{
            background-color: {accent_color};
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            border: none;
        }}

        .stSidebar > div:first-child {{
            background-color: {card_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the theme
def apply_custom_theme(dark):
    if dark:
        bg_color = "#0f0f0f"
        text_color = "#f5f5f5"
        card_color = "#1e1e1e"
        accent_color = "#bb86fc"
        button_bg = "#6200ee"
        status_bg = "#2c2c2c"
        status_text = "#e0e0e0"
    else:
        bg_color = "#f5f5f5"
        text_color = "#000000"
        card_color = "#ffffff"
        accent_color = "#007acc"
        button_bg = "#007acc"
        status_bg = "#eaf4ff"
        status_text = "#003366"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
            color: {accent_color} !important;
        }}

        .css-18e3th9 {{
            background-color: {card_color} !important;
            color: {text_color} !important;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .stSidebar > div:first-child {{
            background-color: {card_color};
        }}

        /* Buttons like Predict & Download Report */
        .stButton>button {{
            background-color: {button_bg} !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.6em 1.2em !important;
            font-weight: bold !important;
            border: none !important;
        }}

        /* Prediction Messages: Analyzing / Complete etc */
        .stAlert {{
            background-color: {status_bg} !important;
            color: {status_text} !important;
            border-radius: 10px;
            padding: 1em;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}

        /* Optional: Markdown-based status styling */
        .status-box {{
            background-color: {status_bg};
            color: {status_text};
            padding: 0.8em;
            border-radius: 8px;
            font-weight: bold;
            margin-top: 1em;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Now add your app content


dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
apply_custom_theme(dark_mode)

# Titles
st.markdown("<h1>🧠 Brain Tumor Detection And Classification App</h1>", unsafe_allow_html=True)
st.markdown("<h4>Upload an MRI Image to Predict Tumor Type</h4>", unsafe_allow_html=True)

# Download the model from Google Drive if not already present
MODEL_PATH = "brain_tumor_resnet50v2_hho_optimized.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "1HrCGlQi3ViiyfAbPaS7JOhZhujLzWBZ0"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

@st.cache_resource
def load_model_hho():
    return load_model(MODEL_PATH)

model = load_model_hho()


from PIL import Image
import streamlit as st

# File Uploader
uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict Button
    if st.button("Predict"):
        st.info("🔍 Analyzing image...")

        # Preprocess and Predict
        image_array = preprocess_image(image)
        result = make_prediction(model, image_array)

        # Show Prediction Results
        st.success("✅ Prediction Complete!")
        st.markdown(f"*Tumor Detected:* {'Yes' if result['has_tumor'] else 'No'}")

        if result['has_tumor']:
            st.markdown(f"*Tumor Type:* {result['tumor_type'].title()}")

        # Generate and Download Report
        report_file = generate_pdf_report(image, result)
        with open(report_file, "rb") as f:
            st.download_button("📄 Download Report", f, file_name="report.pdf")

else:
    st.warning("⚠ Please upload an image file to continue.")