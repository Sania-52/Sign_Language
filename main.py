import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import subprocess
import numpy as np
import random
from PIL import Image
import warnings
from deep_translator import GoogleTranslator
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import pandas as pd
import datetime
import math

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow") 
st.set_page_config(
    page_title="S . I . G . N .",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)
languages = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Bengali (বাংলা)": "bn",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Malayalam (മലയാളം)": "ml",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Urdu (اردو)": "ur",
    "Odia (ଓଡ଼ିଆ)": "or",
    "French": "fr",
    "Spanish": "es",
    "Chinese (中文)": "zh-CN",
    "Arabic (العربية)": "ar"
}
selected_lang = st.selectbox("🌍 Choose Language ", list(languages.keys()))
translator = GoogleTranslator(source="auto", target=languages[selected_lang])
def translate_text(text):
    return translator.translate(text)



# LINK TO CSS FILE               LINK TO CSS FILE                 LINK TO CSS FILE
def load_css(file_path):
    with open(file_path, "r") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
load_css("style.css")

#            TITLE                TITLE                TITLE 
translated_text = translate_text("🤟 S.I.G.N - Sign Interpretation and Gesture Navigation 👐🏻")
st.markdown(
    f"<div class='center-text'>{translated_text}</div>",
    unsafe_allow_html=True
)
#           SUBTITLE             SUBTITLE              SUBTITLE 
st.markdown(f"<div class='sub-header'>{translate_text('Bridging Communication Barriers with AI')}</div>", unsafe_allow_html=True)

#           TEXT-BOX & GIF                TEXT-BOX & GIF             TEXT-BOX & GIF
st.markdown(
    """
    <style>
    .info-text {
        font-family: Arial, sans-serif;
        color: #333;
        background-color: #f4f4f9;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .info-text ul {
        list-style-type: none;
        padding-left: 0;
    }
    .info-text li {
        margin: 10px 0;
    }
    .info-text .gif-container {
        margin-left: 20px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown(f"""
<div class='info-text'>
    <div class='text-content'>
        <p><strong>{translate_text("Project Overview:")}</strong></p>
        <p>{translate_text("S.I.G.N is an AI-powered system designed to interpret sign language gestures in real time.")}</p>
        <ul>
            <li>🤖 {translate_text("AI-based gesture recognition")}</li>
            <li>🖐️ {translate_text("Accurate sign language interpretation")}</li>
            <li>🔊 {translate_text("Voice output for accessibility")}</li>
        </ul>
    </div>
    <div class='gif-container'>
        <img src="https://media1.tenor.com/m/En89xnROufoAAAAC/yes-no.gif" width="200" />
    </div>
</div>
""", unsafe_allow_html=True)


#         LOGO           LOGO          LOGO
st.markdown("<hr style='border: 3px solid #4CAF50; margin-top: 50px; margin-bottom: 50px;'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    st.image("logo.jpg", caption=translate_text("Empowering Voices: Celebrating Indian Sign Language"), use_container_width=True)

#      INFO              INFO             INFO
with col2:
    st.markdown(f"<h1 style='text-align: center; font-size: 30px;'>🙋🏻‍♀️{translate_text('Our S.I.G.N. works with 👇..')}</h1>",unsafe_allow_html=True)
    st.markdown(
    f"""
    <div style='font-size:20px;'>
    1. <b><u>{translate_text("Inclusivity and Communication")}</u>:</b> {translate_text("Indian Sign Language (ISL) bridges the communication gap for the deaf and hard-of-hearing community, fostering inclusivity in education, workplaces, and social interactions.")} <br>
    2. <b><u>{translate_text("Cultural Identity")}</u>:</b> {translate_text("ISL is a rich, visual language that represents India's diverse culture and enhances accessibility for millions across the country.")} <br>
    3. <b><u>{translate_text("Learning Platform")}</u>:</b> {translate_text("This website provides help individuals learn and understand ISL, making communication more accessible for both hearing and non-hearing individuals.")} <br>
    4. <b><u>{translate_text("Awareness and Advocacy")}</u>:</b> {translate_text("By promoting ISL, this platform empowers communities to embrace sign language, encouraging broader societal awareness and inclusiveness.")} 
    </div>
    """,
    unsafe_allow_html=True
)

#        BUTTONS              BUTTONS              BUTTONS

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        detector = HandDetector(maxHands=1)
        classifier = Classifier(
            "../../Model/keras_model.h5",
            "../../Model/labels.txt"
        )
        return detector, classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None
    
    # Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

    # Load model
detector, classifier = load_model()

# Camera settings
imgSize = 50
offset = 5
# Labels
with open("../../Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
# Create placeholders
frame_placeholder = st.empty()
table_placeholder = st.empty()

st.markdown("<hr style='border: 3px solid #4CAF50; margin-top: 50px; margin-bottom: 50px;'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:# Camera toggle button
    if st.button("Start Camera" if not st.session_state.running else "Stop Camera"):
        st.session_state.running = not st.session_state.running
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 50px;'>
            <a href='#' style='
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 20px 34px;
                text-align: center;
                text-decoration: none;
                font-size: 23px;
                border-radius: 8px;'
            >
                {translate_text("Start Camera 📷")}
            </a>
            <p style='font-size: 20px; color: white; margin-top: 5px;'>{translate_text("Turn on your camera to detect gestures 🟩")}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div style='text-align: center;margin-top: 50px;'>
            <a href='#' style='
                display: inline-block;
                background-color: #2196F3;
                color: white;
                padding: 20px 34px;
                text-align: center;
                text-decoration: none;
                font-size: 23px;
                border-radius: 8px;'
            >
                {translate_text("Enter Words ✍🏻")}
            </a>
            <p style='font-size: 20px; color: white; margin-top: 5px;'>{translate_text("Words to Sign Language Generation 🔤✋")}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    "<p style='text-align: center; font-size: 24px; margin-top: 50px; margin-bottom: 50px;'>✨🌟💫✨🌟💫✨🌟💫✨🌟💫✨🌟💫✨🌟💫</p>", 
    unsafe_allow_html=True
)

# Main loop
if st.session_state.running:
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera!")
            st.session_state.running = False
        else:
            while st.session_state.running:
                success, img = cap.read()
                if not success:
                    st.error("Failed to read from camera!")
                    break

                # Detect hands
                hands, img = detector.findHands(img)
                
                if hands:
                    hand = hands[0]  # Get the first hand detected
                    x, y, w, h = hand['bbox']
                    
                    # Ensure bbox coordinates are within image bounds
                    y_start = max(y - offset, 0)
                    y_end = min(y + h + offset, img.shape[0])
                    x_start = max(x - offset, 0)
                    x_end = min(x + w + offset, img.shape[1])
                    
                    imgCrop = img[y_start:y_end, x_start:x_end]
                    
                    if imgCrop.size > 0:  # Check if crop is valid
                        # Prepare white background
                        aspectRatio = h / w
                        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                        
                        # Resize and center the image
                        if aspectRatio > 1:
                            k = imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                            wGap = math.ceil((imgSize - wCal) / 2)
                            imgWhite[:, wGap:wGap + wCal] = imgResize
                        else:
                            k = imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hGap + hCal, :] = imgResize
                        
                        # Get prediction
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        confidence = prediction[index]
                        label = labels[index]
                        
                        # Draw on image
                        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 0, 255), 2)
                        cv2.putText(img, f"{label} ({confidence:.2%})", 
                                  (x_start, y_start - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Store prediction
                        st.session_state.predictions.append({
                            'timestamp': datetime.datetime.now(),
                            'label': label,
                            'confidence': confidence
                        })
                
                # Convert BGR to RGB for Streamlit
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(img_rgb, channels="RGB", use_column_width=True)
                
                # Update predictions table
                if st.session_state.predictions:
                    df = pd.DataFrame(st.session_state.predictions)
                    table_placeholder.dataframe(
                        df.tail(10).style.format({'confidence': '{:.2%}'}),
                        use_container_width=True
                    )
                
            cap.release()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.running = False
# Display summary when stopped
elif st.session_state.predictions:
    df = pd.DataFrame(st.session_state.predictions)
    st.subheader("Detection History")
    st.dataframe(
        df.style.format({'confidence': '{:.2%}'}),
        use_container_width=True
    )
    
    # Add some statistics
    st.subheader("Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detections", len(df))
    with col2:
        st.metric("Unique Gestures", df['label'].nunique())
    with col3:
        st.metric("Avg. Confidence", f"{df['confidence'].mean():.2%}")
        
    # Add a bar chart of gesture frequencies
    st.subheader("Gesture Frequencies")
    gesture_counts = df['label'].value_counts()
    st.bar_chart(gesture_counts) 

# Define columns
col1, col2 = st.columns([1, 2])

# Column 1: Image of LUVY
with col1:
    luvy_image = Image.open("luvy.jpg")  # Ensure this file exists
    st.markdown(
        '<div style="display: flex; justify-content: center; padding-left: 20px; padding-right: 20px;">',
        unsafe_allow_html=True,
    )
    st.image(luvy_image, caption="LUVY - Your Sign Language Assistant made by Sania", width=400)
    st.markdown("</div>", unsafe_allow_html=True)  # Not necessary but safe

# Column 2: Explanation of "Enter Words"
with col2:
    st.markdown(f"<h1 style='text-align: center; font-size: 30px;'>🙋🏻‍♀️ {st.session_state.get('translated_text', 'How Enter Words Works 📝🔍')}</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size:20px; text-align: left;'>
        - Click the <u><b>Enter Words</b></u> button.<br>
        - A text input box appears where you can type any word.<br>
        - Once you press <u><b>Search🔎</b></u>, LUVY searches her pre-animated sign language database.<br>
        - She then displays a short video clip demonstrating the correct sign for your word!<br><br>
        
        This feature makes learning sign language <b>interactive and accessible</b>. Try it now and let LUVY guide you! 🤟🎥
        </div>
        """,
        unsafe_allow_html=True
    )
















#         SIGN CHART              SIGN CHART                SIGN CHART
st.markdown(
    """
    <style>
    .custom-button {
        background-color: #FF69B4; 
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 15px 30px;
        border: none;
        border-radius: 8px;
        transition: background-color 0.3s;
        display: block;
        text-align: center;
        width: 200px;
        margin: auto;
        margin-top: 50px;
    }

    .custom-button:hover {
        background-color: #E91E63;
    }
    </style>
    """,
    unsafe_allow_html=True
)
if st.markdown(f'<button class="custom-button">{translate_text("Sign Chart")}</button>', unsafe_allow_html=True):

    image_urls = [
        "https://clickamericana.com/wp-content/uploads/Native-American-Indian-sign-language-8-750x1199.jpg",
        "https://i.pinimg.com/originals/92/c4/3e/92c43e70dedb715165ff511d2465471d.jpg",
        "https://www.researchgate.net/publication/370152707/figure/fig1/AS:11431281152211749@1682048251037/ndian-Sign-Language-Alphabets-24.png"
    ]
    cols = st.columns(len(image_urls))
    for col, img in zip(cols, image_urls):
        col.image(img, use_container_width=True)



#                        GAME SECTION                                                            GAME SECTION
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
choices = ["Rock", "Paper", "Scissors"]
#            DETECTION
def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y:
        return "Rock"  #✊🏻
    elif index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y and ring_tip.y < pinky_tip.y:
        return "Scissors"   #✌🏻
    elif index_tip.y < middle_tip.y and ring_tip.y < pinky_tip.y:
        return "Paper"   #🖐🏻
    return "Unknown"
def play_game():
    cap = cv2.VideoCapture(0)
    frames = []  # List to hold the 3 frames
    frame_count = 0
    while cap.isOpened() and frame_count < 3:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks)
                cv2.putText(frame, f"Your Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if gesture != "Unknown":
                    ai_choice = random.choice(choices)
                    result = "Tie!"
                    if (gesture == "Rock" and ai_choice == "Scissors") or \
                       (gesture == "Paper" and ai_choice == "Rock") or \
                       (gesture == "Scissors" and ai_choice == "Paper"):
                        result = "You Win!"
                    elif gesture != ai_choice:
                        result = "AI Wins!"
                    cv2.putText(frame, f"AI's Choice: {ai_choice}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, f"Result: {result}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frames.append(frame)
                frame_count += 1
                if frame_count >= 3:
                    break
        cv2.imshow("Sign Language Game", frame)
        if frame_count >= 3 or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # Display the 3 frames 
    cols = st.columns(3)
    for col, frame in zip(cols, frames):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        col.image(pil_image, use_container_width=True)


#     GAME SECTION UI               GAME SECTION UI               GAME SECTION UI
st.markdown("<hr style='border: 3px solid #4CAF50; margin-top: 50px; margin-bottom: 50px;'>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center;'>{translate_text('✊🏻 Rock, 🖐🏻 Paper, ✌🏻 Scissors with Hand Gestures')}</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"<h3 style='text-align: left;'>{translate_text('🙆🏻‍♂️ Show your hand in front of the camera to play!')}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: left;'>{translate_text('🤖 AI will choose randomly to play against you!')}</h4>", unsafe_allow_html=True)
# Column 2: GIF from a URL
with col2:
    gif_url = "https://media.tenor.com/0uoJBcT1WIwAAAAC/how-are-you-signtime.gif" 
    st.image(gif_url, width=200)
if st.button(translate_text("Start Game")):
    play_game()



#  YOUTUBE LINKS                  YOUTUBE LINKS                  YOUTUBE LINKS    
st.markdown("<hr style='border: 3px solid #4CAF50; margin-top: 50px; margin-bottom: 50px;'>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align: center;'>{translate_text('Want to learn Indian Sign Language (ISL)?')}</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .video-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    st.video("https://youtu.be/Vj_13bdU4dU?si=xJ6sj8dvavj_Fd_g")  
with col2:
    st.video("https://youtu.be/l5YwO0VnqWc?si=oYs1IBVxtf0iCz3Y") 
col3, col4 = st.columns(2)
with col3:
    st.video("https://youtu.be/aOL-yBRQHmM?si=GcEADl7MlGV5N-m_") 
with col4:
    st.video("https://youtu.be/6_u9ocF60gs?si=37fXXkAuNwM_INp2") 



#             COMMUNITY CHAT SECTION                      COMMUNITY CHAT SECTION       
st.markdown("<hr style='border: 3px solid #4CAF50; margin-top: 50px; margin-bottom: 50px;'>", unsafe_allow_html=True)   
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
st.markdown(
    """
    <style>
    .chat-container {
        background-color: #f4f4f4;
        padding: 15px;
        border-radius: 10px;
        max-width: 700px;
        margin: auto;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    .chat-message {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 16px;
        line-height: 1.5;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .user-name {
        font-weight: bold;
        color: #007bff;
    }
    .input-container {
        margin-top: 10px;
        text-align: center;
    }
    .input-box {
        width: 80%;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
        font-size: 16px;
    }
    .send-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        margin-left: 10px;
    }
    .send-button:hover {
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(f"<h2 style='text-align: center;'>{translate_text('💬 Community Chat')}</h2>", unsafe_allow_html=True)
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
user_message = st.text_input(translate_text("Type your message here:"), key="chat_input")
# Submit button
if st.button("Send"):
    if user_message.strip():
        st.session_state.chat_messages.append(user_message)
        st.experimental_rerun() 

















def load_css(file_path):
    with open(file_path, "r") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
 
# Load the CSS file
load_css("style.css")  # Ensure this file is in the same directory as main.py


# Footer HTML
footer = """
    <div class="footer">
        Developed with ❤️ by <a href="https://github.com/Sania-52" target="_blank">Sania</a> | © 2025 All Rights Reserved
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)