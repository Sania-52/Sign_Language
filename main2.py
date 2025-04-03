import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator

# Set page configuration
st.set_page_config(
    page_title="S . I . G . N .",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available languages with Indian languages included
languages = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Marathi (मराठी)": "mr",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Gujarati (ગુજરાતી)": "gu",
    "Bengali (বাংলা)": "bn",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "French": "fr",
    "Spanish": "es",
    "Chinese": "zh-CN",
    "Arabic": "ar"
}

# Language selection
selected_lang = st.selectbox("🌍 Choose Language", list(languages.keys()))
translator = GoogleTranslator(source="auto", target=languages[selected_lang])

# Function to translate text dynamically
def translate_text(text):
    return translator.translate(text)

# Translated UI Texts
title = translate_text("S.I.G.N - Sign Interpretation and Gesture Navigation")
description = translate_text("Bridging Communication Barriers with AI")
button_camera = translate_text("Start Camera 📷")
button_words = translate_text("Enter Words ✍🏻")
sign_chart_button = translate_text("Sign Chart 📜")

# Apply CSS for styling
st.markdown("""
    <style>
        .center-text {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            margin-top: 10px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 50px;
            color: gray;
        }
        .top-right-buttons {
            position: absolute;
            top: 10px;
            right: 20px;
            display: flex;
            gap: 10px;
        }
        .top-right-buttons a {
            text-decoration: none;
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            font-weight: bold;
            display: inline-block;
        }
        .top-right-buttons a:hover {
            background-color: #0056b3;
        }
    </style>
    """, unsafe_allow_html=True)

# Display Translated UI
st.markdown(f"<div class='center-text'>{title}</div>", unsafe_allow_html=True)
st.write(description)

# Button section
col1, col2 = st.columns(2)
with col1:
    st.button(button_camera)
with col2:
    st.button(button_words)

# Sign Chart Button
if st.button(sign_chart_button):
    st.write("### " + translate_text("Sign Language Charts 📜"))
    
    # List of images (Use Google Drive links or direct URLs)
    image_urls = [
        "https://clickamericana.com/wp-content/uploads/Native-American-Indian-sign-language-8-750x1199.jpg",
        "https://i.pinimg.com/originals/92/c4/3e/92c43e70dedb715165ff511d2465471d.jpg",
        "https://www.researchgate.net/publication/370152707/figure/fig1/AS:11431281152211749@1682048251037/ndian-Sign-Language-Alphabets-24.png"
    ]
    
    # Display images in columns
    cols = st.columns(len(image_urls))
    for col, img in zip(cols, image_urls):
        col.image(img, use_container_width=True)

# Image Section
st.subheader(translate_text("Example Images"))

image1 = Image.open("example.jpg").resize((300, 300))
image2 = Image.open("example1.jpg").resize((300, 300))
image3 = Image.open("example2.jpg").resize((300, 300))

col1, col2, col3 = st.columns(3)
with col1:
    st.image(image1, caption=translate_text("Image 1"))
with col2:
    st.image(image2, caption=translate_text("Image 2"))
with col3:
    st.image(image3, caption=translate_text("Image 3"))

# Footer
footer_text = translate_text("Developed with ❤️ by ") + "<a href='https://github.com/Sania-52' target='_blank'>Sania</a> | © 2025 " + translate_text("All Rights Reserved")
st.markdown(f"<div class='footer'>{footer_text}</div>", unsafe_allow_html=True)
