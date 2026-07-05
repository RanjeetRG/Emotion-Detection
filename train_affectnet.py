import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

# =========================================================
# STAGE 2 CONFIG — AFFECTNET FINE-TUNING
# =========================================================
DATA_DIR = "affectnet_data"        # <-- IMPORTANT
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-4                          # LOW LR FOR FINE-TUNING
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("====================================")
print("Using device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("====================================")

# =========================================================
# IMAGE TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# LOAD DATA (WINDOWS SAFE)
# =========================================================
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),   # test used as val
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

class_names = train_dataset.classes
print("Classes:", class_names)

# =========================================================
# CLASS WEIGHTS (CRITICAL FOR AFFECTNET)
# =========================================================
labels = [label for _, label in train_dataset.samples]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# =========================================================
# LOAD FER-PRETRAINED MODEL
# =========================================================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(
    torch.load("emotion_model_fer.pth", map_location=DEVICE)
)

model = model.to(DEVICE)

# =========================================================
# LOSS, OPTIMIZER, SCHEDULER
# =========================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(model.parameters(), lr=LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=3,
    factor=0.5,
    verbose=True
)

# =========================================================
# TRAINING LOOP
# =========================================================
best_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    ):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ---------------- VALIDATION ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    scheduler.step(accuracy)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {running_loss:.4f} | "
        f"Val Accuracy: {accuracy:.4f}"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "emotion_model.pth")
        print("✅ Best FINAL model saved")

print("\n🎉 STAGE 2 COMPLETE")
print("Best Validation Accuracy:", best_accuracy)
