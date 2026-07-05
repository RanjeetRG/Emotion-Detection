# Emotion Detection — Real-Time Facial Expression Recognition

A real-time emotion detection system that reads your webcam feed, finds your face, and tells you what emotion you're showing — all running locally on your machine.

Built with PyTorch and OpenCV. No cloud, no API, just your GPU doing the work.

---

## What it does

- Detects faces in real-time using a pre-trained SSD face detector (Caffe model)
- Classifies the detected face into one of 7 emotions every frame
- Applies temporal smoothing so the label doesn't flicker on every frame
- Displays the result directly on the webcam window

**Emotions detected:** Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise

---

## How it works

The face detector runs first — it's a lightweight SSD model from OpenCV's DNN module that's fast enough to run every frame without lag.

Once a face is found, it gets cropped out and passed through a ResNet18 model that was trained in two stages:

1. **Stage 1** — Pre-trained on FER2013 to learn basic emotion patterns
2. **Stage 2** — Fine-tuned on AffectNet with class balancing (AffectNet is heavily skewed toward neutral/happy, so class weights were used to fix that)

The final prediction isn't just raw argmax — there's a rule-based layer on top that handles edge cases like angry vs. fear confusion, and a 12-frame rolling window that prevents the label from jumping around too much.

---

## Setup

Clone the repo and activate a virtual environment, then install dependencies:

```bash
pip install torch torchvision opencv-python numpy
```

Make sure you have the model files in the root directory:
- `emotion_model.pth` — the trained emotion classifier
- `res10_300x300_ssd_iter_140000.caffemodel` — face detector weights
- `deploy.prototxt` — face detector config

---

## Run

```bash
python detect_emotion.py
```

Press **Q** to quit.

If you have a CUDA-capable GPU, it'll use it automatically. Otherwise falls back to CPU (slightly slower but works fine).

---

## Training your own model

If you want to retrain:

1. Put your dataset in `affectnet_data/train/` and `affectnet_data/test/` — one folder per emotion class
2. Have a base FER-pretrained model saved as `emotion_model_fer.pth`
3. Run:

```bash
python train_affectnet.py
```

Training config (edit at the top of the file):
- Batch size: 64
- Epochs: 15
- Optimizer: AdamW (lr=1e-4)
- Scheduler: ReduceLROnPlateau

The best model (by validation accuracy) gets saved automatically as `emotion_model.pth`.

---

## Project structure

```
affectnet_project/
├── detect_emotion.py              # main script — run this
├── train_affectnet.py             # training script (stage 2)
├── prepare_affectnet.py           # dataset prep utility
├── emotion_model.pth              # trained model weights
├── emotion_model_fer.pth          # FER pre-trained base model
├── deploy.prototxt                # face detector config
├── res10_300x300_ssd_iter_140000.caffemodel   # face detector weights
└── .gitignore
```

---

## Known issues / notes

- The model sometimes confuses **fear** and **angry** — there's logic in the detection script to handle this, but it's not perfect
- **Disgust** rarely triggers in practice since it's underrepresented in most datasets
- Low lighting will significantly hurt face detection accuracy
- Tested on Windows with Python 3.10, PyTorch 2.5, CUDA 12.1

---

## Tech stack

- Python 3.10
- PyTorch 2.5
- OpenCV 4.x (DNN module for face detection)
- torchvision (ResNet18 backbone)
- scikit-learn (class weight computation)
