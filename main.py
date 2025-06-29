# ASL Real-Time Sign Detection (Detection-Only Version)
# This script loads a trained Keras model (asl_model.h5) and uses YOLO for hand detection.
# It runs the webcam and predicts ASL signs in real time.

import cv2
import numpy as np
import string
from collections import deque
import time
from tensorflow.keras.models import load_model
import os

# --- CONFIG ---
MAX_HISTORY = 100
ASL_LETTERS = list(string.ascii_uppercase) + ["del", "nothing", "space"]  # Update if your model has 29 classes
MODEL_PATH = 'asl_model.h5'  # Path to your trained Keras model
MODEL_INPUT_SIZE = (192, 192)  # FIXED: Match training input size
HAND_CFG = 'cross-hands-tiny-prn.cfg'
HAND_WEIGHTS = 'cross-hands-tiny-prn.weights'
HAND_CONFIDENCE = 0.3

# Load the trained Keras model
if os.path.exists(MODEL_PATH):
    asl_model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    asl_model = None
    print(f"WARNING: Model file '{MODEL_PATH}' not found. The recognizer will not work.")

def real_predict(hand_img):
    if asl_model is None:
        return '?'  # Model not loaded
    img = cv2.resize(hand_img, MODEL_INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    pred = asl_model.predict(img, verbose=0)
    idx = np.argmax(pred)
    confidence = np.max(pred)
    if confidence > 0.3:
        return ASL_LETTERS[idx]
    else:
        return "nothing"

class ASLHandDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet(HAND_CFG, HAND_WEIGHTS)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.pred_history = deque(maxlen=MAX_HISTORY)
        self.current_text = ""
        self.last_letter = ""
        self.last_time = time.time()
        self.all_predictions = []
        self.letter_buffer = []
        self.buffer_start_time = time.time()
        self.buffer_interval = 3.0  # seconds to wait before adding a letter

    def detect_hands(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        ln = self.net.getLayerNames()
        out_layers = self.net.getUnconnectedOutLayers()
        if len(out_layers.shape) == 2:
            ln = [ln[i[0] - 1] for i in out_layers]
        else:
            ln = [ln[i - 1] for i in out_layers]
        detections = self.net.forward(ln)
        hands = []
        h, w = frame.shape[:2]
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > HAND_CONFIDENCE:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x1 = int(centerX - (width / 2))
                    y1 = int(centerY - (height / 2))
                    x2 = int(centerX + (width / 2))
                    y2 = int(centerY + (height / 2))
                    hands.append((x1, y1, x2, y2, float(confidence)))
        return hands

    def process_predictions(self):
        result = []
        prev = None
        for letter in self.all_predictions:
            if letter == prev:
                continue
            prev = letter
            if letter == "nothing" or letter is None:
                continue
            elif letter == "space":
                result.append(" ")
            else:
                result.append(letter)
        return "".join(result)

    def run(self):
        cap = cv2.VideoCapture(0)
        print("Starting real-time ASL hand detector. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            hands = self.detect_hands(frame)
            letter = None
            for (x1, y1, x2, y2, conf) in hands:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                hand_img = frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])]
                if hand_img.size > 0:
                    letter = real_predict(hand_img)
                    self.pred_history.append(letter)
                    self.all_predictions.append(letter)
                    if letter is not None and letter != "nothing":
                        self.letter_buffer.append(letter)
                    break
            else:
                self.pred_history.append(None)
                self.all_predictions.append(None)
            now = time.time()
            if now - self.buffer_start_time > self.buffer_interval:
                if self.letter_buffer:
                    filtered = [l for l in self.letter_buffer if l is not None]
                    if filtered:
                        most_common = max(set(filtered), key=filtered.count)
                        self.current_text += most_common
                        self.last_letter = most_common
                self.letter_buffer = []
                self.buffer_start_time = now
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Prediction: {letter if letter else '-'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Text: {self.current_text}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("YOLO Hand Detection ASL", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('c'):
                self.current_text = ""
        cap.release()
        cv2.destroyAllWindows()
        final_result = self.process_predictions()
        print("\nFinal ASL result:", final_result)

if __name__ == "__main__":
    detector = ASLHandDetector()
    detector.run()
