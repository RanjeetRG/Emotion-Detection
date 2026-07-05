import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from collections import deque

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "emotion_model.pth"
PROTOTXT = "deploy.prototxt"
CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"

CONF_FACE = 0.6

# MUST MATCH TRAINING ORDER
emotions = ['angery', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================================================
# LOAD FACE DETECTOR
# =========================================================
face_net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

# =========================================================
# LOAD EMOTION MODEL
# =========================================================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(emotions))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================================================
# IMAGE TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# TEMPORAL SMOOTHING
# =========================================================
emotion_window = deque(maxlen=12)  # Balanced window — stable but responsive
sticky_emotion = "neutral"          # Holds the last confirmed emotion
STABILITY_THRESHOLD = 0.50         # New emotion must appear in 50% of frames to switch

# =========================================================
# START WEBCAM
# =========================================================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < CONF_FACE:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # -------------------------------------------------
        # EMOTION PREDICTION
        # -------------------------------------------------
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = transform(face_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(face_tensor), dim=1)[0]

        angry_p   = probs[emotions.index("angery")].item()
        fear_p    = probs[emotions.index("fear")].item()
        happy_p   = probs[emotions.index("happy")].item()
        neutral_p = probs[emotions.index("neutral")].item()
        sad_p     = probs[emotions.index("sad")].item()
        surprise_p= probs[emotions.index("surprise")].item()

        # =================================================
        # FINAL HUMAN-LIKE DECISION LOGIC
        # =================================================

        # 😲 SURPRISE (very strict)
        if surprise_p > 0.55 and surprise_p > happy_p + 0.10:
            emotion = "surprise"

        # 🙂 HAPPY (dominant over surprise & neutral)
        elif happy_p > 0.35 and happy_p > neutral_p + 0.05:
            emotion = "happy"

        # 😠 ANGER dominates fear unless fear is VERY strong
        elif angry_p > 0.30 and angry_p > fear_p - 0.05:
            emotion = "angery"

        # 😨 FEAR only if clearly dominant
        elif fear_p > 0.45 and fear_p > angry_p + 0.10:
            emotion = "fear"

        # 😐 NEUTRAL (sad mapped here)
        elif neutral_p > 0.25 or sad_p > 0.35:
            emotion = "neutral"

        else:
            emotion = emotions[torch.argmax(probs).item()]
            if emotion in ["sad", "fear"]:
                emotion = "neutral"

        # -------------------------------------------------
        # TEMPORAL SMOOTHING
        # -------------------------------------------------
        emotion_window.append(emotion)

        # Only switch displayed emotion if dominant one clears the stability threshold
        dominant = max(set(emotion_window), key=emotion_window.count)
        dominant_ratio = emotion_window.count(dominant) / len(emotion_window)

        if dominant_ratio >= STABILITY_THRESHOLD:
            sticky_emotion = dominant

        final_emotion = sticky_emotion

        # -------------------------------------------------
        # DRAW
        # -------------------------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            final_emotion.upper(),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detection (FINAL)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
