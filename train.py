import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# -----------------------------
# SETTINGS
# -----------------------------
DATA_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.pth")

IMAGE_SIZE = 128
BATCH_SIZE = 32
MAX_IMAGES = 2000

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

# Save class names BEFORE random_split
classes = dataset.classes

print("\nClasses:")
for i, name in enumerate(classes):
    print(i, "->", name)

print("\nOriginal images:", len(dataset))

# -----------------------------
# USE SMALL DATASET FOR TEST
# -----------------------------
if len(dataset) > MAX_IMAGES:
    dataset, _ = random_split(
        dataset,
        [MAX_IMAGES, len(dataset) - MAX_IMAGES]
    )

print("Images used for test:", len(dataset))

# -----------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# -----------------------------
# DATA LOADERS
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# LOAD RESNET18
# -----------------------------
print("\nLoading ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze ResNet layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model = model.to(device)

# -----------------------------
# LOSS & OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

# -----------------------------
# FAST TEST TRAINING
# -----------------------------
print("\nStarting FAST TEST TRAINING...\n")

model.train()

correct = 0
total = 0
running_loss = 0.0

for images, labels in train_loader:

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    _, predicted = torch.max(outputs, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

# -----------------------------
# RESULT
# -----------------------------
print("\n==============================")
print("Training finished!")
print("Training Accuracy:", round(accuracy, 2), "%")
print("==============================")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes
    },
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)