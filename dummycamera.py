import streamlit as st
import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

# Function to run the camera code
def run_camera():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)

    offset = 20
    imgSize = 300
    counter = 0
    motion_threshold = 8000
    prev_frame = None

    folder = "C:/Users/Dell/Documents/sahil project/Sign-Language-detection/Data/I have a Doubt"

    while True:
        success, img = cap.read()
        if not success:
            continue

        hands, img = detector.findHands(img)
        motion_detected = False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            motion_value = np.sum(thresh)
            if motion_value > motion_threshold:
                motion_detected = True
        prev_frame = gray

        if hands and motion_detected:
            for i, hand in enumerate(hands):
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[max(y-offset, 0):min(y+h+offset, img.shape[0]),
                              max(x-offset, 0):min(x+w+offset, img.shape[1])]

                if imgCrop.size == 0:
                    continue

                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                counter += 1
                cv2.imwrite(f'{folder}/Image_{counter}_{time.time()}.jpg', imgWhite)
                print(f"Captured {counter} images")

                cv2.imshow(f'Hand_{i+1}', imgWhite)

        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit Interface
st.title("Hand Detection System")

if st.button("Start Camera"):
    run_camera()
