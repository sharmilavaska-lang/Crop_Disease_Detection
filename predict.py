import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "model/crop_disease_model.pth"
IMAGE_PATH = "test_leaf.jpg"
IMAGE_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]

print("Classes loaded:", len(classes))

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, prediction = torch.max(probabilities, 1)

predicted_class = classes[prediction.item()]
confidence_value = confidence.item() * 100

print("\n==============================")
print("AI PREDICTION")
print("==============================")
print("Disease:", predicted_class)
print("Confidence:", round(confidence_value, 2), "%")
print("==============================")