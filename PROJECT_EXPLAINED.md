# How This Project Was Built — A Full Technical Breakdown

This document covers everything about the emotion detection system — the algorithms, the math behind them, why certain choices were made, and how the whole thing fits together. Written so someone with basic ML knowledge can follow along.

---

## The Big Picture

At a high level the system does two things:

1. **Find the face** in a webcam frame
2. **Classify the emotion** shown on that face

These are two separate models chained together. The face detector runs first, then passes a cropped face image to the emotion classifier. Both run every frame in real-time.

---

## Part 1 — Face Detection (SSD + Caffe)

### What model is used

The face detector is a pre-trained **Single Shot MultiBox Detector (SSD)** loaded via OpenCV's DNN module. The weights come from a Caffe model trained on face data — `res10_300x300_ssd_iter_140000.caffemodel`.

We didn't train this from scratch. It's a frozen, ready-to-use face detector.

### How SSD works

Traditional object detectors like Faster R-CNN use two stages: first propose regions that might contain objects, then classify those regions. SSD skips the region proposal step and does everything in a single pass — hence "Single Shot."

The image is resized to 300×300 and passed through a CNN. At different layers of the network (different scales), the model predicts:
- Whether a face exists in each grid cell
- A bounding box offset to refine where exactly the face is

**The confidence score formula:**

```
confidence = sigmoid(raw_score)
```

We only keep detections where `confidence > 0.6`. Anything below that is treated as background.

**Bounding box decoding:**

The model doesn't output pixel coordinates directly. It outputs offsets relative to fixed anchor boxes. The final box is:

```
x_center = anchor_x + offset_x * anchor_width
y_center = anchor_y + offset_y * anchor_height
width     = anchor_width  * exp(scale_w)
height    = anchor_height * exp(scale_h)
```

These are then scaled back to the original image resolution.

### Why OpenCV DNN and not a Python face library

Speed. Libraries like `face_recognition` or `dlib` are accurate but slow for real-time. OpenCV's DNN module runs inference in C++ under the hood, making it fast enough to run every frame without dropping frames.

---

## Part 2 — The Emotion Classifier (ResNet18)

### Architecture

The classifier is a **ResNet18** — a convolutional neural network with 18 layers that uses residual (skip) connections.

#### What a residual connection does

In a normal CNN, each layer's output is passed to the next:

```
output = F(input)
```

In ResNet, the input is added directly to the output of a block:

```
output = F(input) + input
```

This is called a **skip connection** or **residual connection**. It solves a real problem — when you stack many layers, gradients get tiny during backpropagation and the early layers stop learning (vanishing gradient problem). Adding the raw input back gives the gradient a shortcut path to flow through.

#### Layer breakdown (simplified)

```
Input (224×224×3)
    ↓
Conv 7×7, stride 2  →  112×112×64
Max Pool            →   56×56×64
4× Residual Blocks  →   ...
    ↓
Global Average Pool →  1×1×512
    ↓
Fully Connected     →  7 outputs (one per emotion)
```

The final layer outputs 7 raw numbers called **logits** — one for each emotion class.

### From logits to probabilities — Softmax

Logits are unbounded. To turn them into probabilities that sum to 1, we apply **softmax**:

```
P(class_i) = exp(z_i) / Σ exp(z_j)   for j = 0..6
```

Where `z_i` is the raw output for class i. The exponential ensures all values are positive, and dividing by the sum normalizes them.

Example:
```
Raw logits:  [2.1, -0.5, 0.3, 3.2, 1.1, -1.0, 0.8]
After softmax: [0.14, 0.01, 0.02, 0.55, 0.07, 0.01, 0.05]
               angry disgust fear happy neutral sad surprise
```
Here the model is 55% confident the emotion is "happy."

---

## Part 3 — Training

Training was done in two stages. This is called **transfer learning with fine-tuning**.

### Why two stages

Training from scratch on emotion data alone doesn't work well because:
- Emotion datasets are small relative to what a CNN needs to learn good visual features
- Starting from random weights wastes a lot of compute learning basic things like edges and textures

So instead:
- **Stage 1**: Train on FER2013 (a large but noisily labeled dataset) to learn general emotion features
- **Stage 2**: Fine-tune on AffectNet (higher quality, more in-the-wild images) with a low learning rate

### Loss Function — Cross-Entropy Loss

During training, the model's job is to maximize the probability it assigns to the correct class. The loss function measures how wrong it is:

```
L = -log(P(correct_class))
```

If the model is 90% confident in the right answer, loss = -log(0.9) = 0.046 (small, good)
If the model is 10% confident, loss = -log(0.1) = 2.3 (large, bad)

For a full batch of samples:

```
L_total = -(1/N) * Σ log(P(correct_class_i))
```

This is called **Categorical Cross-Entropy Loss**.

### Handling class imbalance — weighted loss

AffectNet has a massive imbalance: there are way more "neutral" and "happy" images than "disgust" or "fear." If you train on this naively, the model learns to just predict neutral/happy all the time.

Fix: give rarer classes higher weights in the loss function.

```
L_weighted = -Σ w_i * log(P(correct_class_i))
```

Where `w_i` is higher for rare classes. The weights are computed using sklearn's `compute_class_weight("balanced", ...)` which sets:

```
w_i = total_samples / (num_classes * count_of_class_i)
```

So if "disgust" has 1000 samples and "neutral" has 50000, disgust gets ~50x higher weight.

### Optimizer — AdamW

AdamW is the optimizer used to update model weights. It combines two ideas:

**Momentum** — instead of updating weights using just the current gradient, use an exponential moving average of past gradients:
```
m_t = β1 * m_{t-1} + (1 - β1) * g_t
```

**Adaptive learning rates** — each weight gets its own learning rate based on how large its gradients have been:
```
v_t = β2 * v_{t-1} + (1 - β2) * g_t²
```

**Final update rule:**
```
θ_t = θ_{t-1} - lr * m_t / (√v_t + ε) - lr * λ * θ_{t-1}
```

The last term (`λ * θ_{t-1}`) is **weight decay** — it regularizes the model by penalizing large weights. This is the "W" in AdamW.

Learning rate used: `1e-4` (small, since we're fine-tuning an already decent model, not training from scratch).

### Learning Rate Scheduler

```python
ReduceLROnPlateau(mode="max", patience=3, factor=0.5)
```

If validation accuracy doesn't improve for 3 consecutive epochs, the learning rate is cut in half. This prevents the model from overshooting good minima as training progresses.

### Data Augmentation

To prevent overfitting and make the model robust to real-world variation:

```
RandomHorizontalFlip()      — face emotions are symmetric left-right
RandomRotation(10°)         — people don't always face cameras straight
ColorJitter(brightness, contrast) — different lighting conditions
```

These are applied only during training, not validation.

---

## Part 4 — Real-Time Inference Logic

### The decision layer

Raw softmax probabilities aren't directly used to pick the final label. There's a rule-based layer on top that handles known confusion cases:

```python
if surprise_p > 0.55 and surprise_p > happy_p + 0.10:
    emotion = "surprise"
elif happy_p > 0.35 and happy_p > neutral_p + 0.05:
    emotion = "happy"
elif angry_p > 0.30 and angry_p > fear_p - 0.05:
    emotion = "angry"
elif fear_p > 0.45 and fear_p > angry_p + 0.10:
    emotion = "fear"
elif neutral_p > 0.25 or sad_p > 0.35:
    emotion = "neutral"
```

Why? Because angry and fear look visually similar (furrowed brows, wide eyes). The model often assigns high probability to both simultaneously. These rules break the tie in a human-intuitive way.

### Temporal smoothing

Raw per-frame predictions jump around because each frame is processed independently. If someone shifts their head slightly, the prediction changes.

Fix: maintain a rolling window of the last 12 predictions using a `deque`:

```
window = [happy, happy, neutral, happy, happy, happy, happy, happy, neutral, happy, happy, happy]
```

The dominant emotion in the window must appear in at least 50% of frames to become the displayed label:

```
dominant_ratio = count(dominant) / len(window)
if dominant_ratio >= 0.50:
    display = dominant
```

This makes the label stable and human-readable without being laggy.

---

## Part 5 — Where This Can Be Used

### Mental health monitoring
Track emotional state over time during a therapy session or daily check-in. Not for diagnosis, but as one data signal.

### E-learning / online exams
Detect confusion, frustration, or disengagement in students during online classes. Alert teachers or adjust pacing.

### Customer feedback
Retail or UX research — see how people react to a product demo, ad, or interface without them having to fill out a survey.

### Driver monitoring
Detect drowsiness (neutral → drooping face) or stress in drivers. Already being used in some automotive systems.

### Gaming / VR
Map player facial expressions to in-game character animations. Makes avatars more expressive without manual input.

### HR / interviews (controversial)
Some companies have tried this for job screening. It's ethically questionable and largely inaccurate out of context — mentioning it here only for completeness.

---

## Limitations worth knowing

- **Lighting** is the biggest enemy. Poor lighting breaks the face detector before it even reaches the classifier.
- **Cultural differences** — facial expression of emotions varies across cultures. Training data is mostly Western faces.
- **Context blindness** — the model sees a 224×224 crop of a face with zero context. A "fear" face and a "surprise" face look almost identical.
- **Single frame analysis** — emotions unfold over time. A single frame misses the full picture. Video-based models (like those trained on AffWild2) do better.
- **Disgust** almost never fires correctly in practice — it's underrepresented in training data and visually subtle.

---

## Summary

| Component | What it is | Formula / key idea |
|---|---|---|
| Face detector | SSD (Caffe) | Single-pass CNN, confidence threshold = 0.6 |
| Backbone | ResNet18 | Residual connections: `output = F(x) + x` |
| Output activation | Softmax | `P(i) = exp(z_i) / Σ exp(z_j)` |
| Loss function | Weighted Cross-Entropy | `L = -Σ w_i * log(P_i)` |
| Optimizer | AdamW | Momentum + adaptive LR + weight decay |
| Imbalance fix | Class weights | `w_i = N / (K * n_i)` |
| Stability | Temporal smoothing | 12-frame rolling window, 50% threshold |
| Edge cases | Rule-based override | Threshold-based angry/fear/neutral disambiguation |
