import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pygame
import time
from collections import deque


IMAGE_SIZE = 224
CONF_THRESHOLD = 0.35      
SMOOTHING_WINDOW = 10      
AUDIO_COOLDOWN = 1.5       


pygame.mixer.init()

audio_map = {
    "selamat": "selamat.mp3",
    "berjuang": "berjuang.mp3",
    "sukses": "sukses.mp3",
    "hidup": "hidup.mp3"
}

last_audio_time = 0
last_audio_label = None


model = keras.models.load_model("keras_model.h5", compile=False)


labels = {}
with open("labels.txt", "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(" ")
        labels[int(idx)] = name


prediction_history = deque(maxlen=SMOOTHING_WINDOW)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.flip(frame, 1)

    
    img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

   
    preds = model.predict(img, verbose=0)[0]
    prediction_history.append(preds)

    avg_preds = np.mean(prediction_history, axis=0)

    idx = np.argmax(avg_preds)
    confidence = avg_preds[idx]
    label = labels[idx]
    confidence_percent = confidence * 100

    
    color = (0, 255, 0) if confidence > CONF_THRESHOLD else (0, 0, 255)
    cv2.putText(
        frame,
        f"{label} : {confidence_percent:.1f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    
    y_offset = 80
    for i in range(len(avg_preds)):
        txt = f"{labels[i]} : {avg_preds[i]*100:.1f}%"
        cv2.putText(
            frame,
            txt,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
        y_offset += 25

    
    now = time.time()

    if confidence > CONF_THRESHOLD and label != "diam":
        if label != last_audio_label and (now - last_audio_time) > AUDIO_COOLDOWN:
            if label in audio_map:
                pygame.mixer.music.load(audio_map[label])
                pygame.mixer.music.play()
                last_audio_label = label
                last_audio_time = now

    
    cv2.imshow("Teachable Machine - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
