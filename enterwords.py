import streamlit as st
from PIL import Image
import random
import os

def get_video_for_word(word):
    """Fetches a pre-animated sign language video from local storage."""
    video_folder = "Sign_Language/videos"

    videos = {
        "how are you": os.path.join(video_folder, "how_are_you.mp4"),
        "have a good day": os.path.join(video_folder, "have_a-good_day.mp4"),
        "oh my god": os.path.join(video_folder, "oh_my_god.mp4"),
        "love": os.path.join(video_folder, "love.mp4"),
        "i am sorry": os.path.join(video_folder, "i_am_sorry.mp4"),
        "i am excited": os.path.join(video_folder, "i_am_excited.mp4")
    }
    
    return videos.get(word.lower(), os.path.join(video_folder, "notfound.mp4"))

# Streamlit UI
st.title("LUVY - Enter Words to Generate Sign Language")

st.markdown(
    """
    ### How It Works 📝🔍
    - Enter a word in the input box below.
    - Click **Search** to see LUVY display the sign language animation.
    - If the word is not available, a default 'not found' message appears.
    """
)

# User Input
#user_word = st.text_input("Enter a word to see its sign language representation:")

word = st.text_input("Enter a word:")
if st.button("Search"):
    video_path = get_video_for_word(word)

    st.video(video_path)
