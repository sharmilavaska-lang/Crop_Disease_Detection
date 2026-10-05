from flask import Flask, render_template, request
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ==============================
# SETTINGS
# ==============================

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/crop_disease_model.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==============================
# DEVICE
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================
# LOAD MODEL
# ==============================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]

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


# ==============================
# IMAGE TRANSFORM
# ==============================

IMAGE_SIZE = 128

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==============================
# DISEASE INFORMATION
# ==============================

disease_info = {

    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "scientific_name": "Xanthomonas spp.",
        "status": "Disease Detected",
        "severity": "Medium",
        "cause": "Bacterial infection, often encouraged by wet conditions.",
        "symptoms": "Small dark spots may appear on leaves and can spread over time.",
        "action": "Remove badly affected leaves and keep the plant area clean.",
        "treatment": "Follow local agricultural guidance for suitable treatment. Avoid unnecessary chemical use.",
        "prevention": "Avoid overhead watering and maintain good airflow around plants."
    },

    "Tomato___Early_blight": {
        "name": "Early Blight",
        "scientific_name": "Alternaria solani",
        "status": "Disease Detected",
        "severity": "Medium",
        "cause": "Fungal infection commonly associated with warm and humid conditions.",
        "symptoms": "Brown circular spots with darker rings may develop on older leaves.",
        "action": "Remove severely affected leaves and improve airflow.",
        "treatment": "Follow local agricultural guidance for appropriate fungicide treatment if necessary.",
        "prevention": "Avoid prolonged leaf wetness and keep the growing area clean."
    },

    "Tomato___healthy": {
        "name": "Healthy Leaf",
        "scientific_name": "No disease detected",
        "status": "Healthy",
        "severity": "Low",
        "cause": "No visible disease pattern was detected by the AI model.",
        "symptoms": "No major disease symptoms detected.",
        "action": "Continue normal plant care and monitor the crop regularly.",
        "treatment": "No disease treatment is indicated based on this prediction.",
        "prevention": "Maintain good sunlight, airflow, watering and regular crop monitoring."
    },

    "Tomato___Late_blight": {
        "name": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "status": "Disease Detected",
        "severity": "High",
        "cause": "Fungal-like pathogen that spreads rapidly in cool, wet and humid conditions.",
        "symptoms": "Dark irregular lesions may develop on tomato leaves.",
        "action": "Separate affected plants where possible and inspect nearby plants for similar symptoms.",
        "treatment": "Seek local agricultural guidance for suitable treatment, especially when symptoms spread quickly.",
        "prevention": "Improve airflow and avoid prolonged leaf wetness."
    },

    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "scientific_name": "Passalora fulva",
        "status": "Disease Detected",
        "severity": "Medium",
        "cause": "Fungal infection favored by high humidity.",
        "symptoms": "Yellowish patches may appear on the upper surface of leaves with mold growth underneath.",
        "action": "Remove severely affected leaves and improve ventilation.",
        "treatment": "Follow local agricultural guidance for appropriate fungal disease management.",
        "prevention": "Reduce humidity around plants and avoid wetting the leaves."
    },

    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "status": "Disease Detected",
        "severity": "Medium",
        "cause": "Fungal disease that spreads through moisture and infected plant material.",
        "symptoms": "Small circular spots with darker edges can appear on leaves.",
        "action": "Remove affected leaves and keep fallen plant material away from the crop.",
        "treatment": "Follow local agricultural guidance for suitable disease management.",
        "prevention": "Avoid overhead watering and maintain good spacing between plants."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Spider Mites",
        "scientific_name": "Tetranychus urticae",
        "status": "Pest Detected",
        "severity": "Medium",
        "cause": "Spider mite infestation, often favored by hot and dry conditions.",
        "symptoms": "Tiny yellow or pale spots and fine webbing may appear on leaves.",
        "action": "Inspect the underside of leaves and isolate heavily affected plants where possible.",
        "treatment": "Seek local agricultural guidance for suitable pest management.",
        "prevention": "Monitor plants regularly and maintain appropriate moisture."
    },

    "Tomato___Target_Spot": {
        "name": "Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "status": "Disease Detected",
        "severity": "Medium",
        "cause": "Fungal infection that can spread under warm and humid conditions.",
        "symptoms": "Circular brown spots with concentric rings may appear on leaves.",
        "action": "Remove heavily affected leaves and improve airflow.",
        "treatment": "Follow local agricultural guidance for suitable disease management.",
        "prevention": "Avoid prolonged leaf wetness and keep the crop area clean."
    },

    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "scientific_name": "Tomato mosaic virus",
        "status": "Virus Detected",
        "severity": "High",
        "cause": "Viral infection that can spread through contaminated plant material and handling.",
        "symptoms": "Mosaic-like light and dark green patterns may appear on leaves.",
        "action": "Remove severely affected plants and avoid spreading contaminated plant material.",
        "treatment": "There is no simple curative treatment for an infected plant. Seek agricultural guidance.",
        "prevention": "Use clean tools and healthy planting material."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "scientific_name": "TYLCV",
        "status": "Virus Detected",
        "severity": "High",
        "cause": "Viral disease commonly spread by whiteflies.",
        "symptoms": "Leaves may curl upward, become smaller and show yellowing.",
        "action": "Inspect nearby plants and monitor for whitefly activity.",
        "treatment": "Infected plants may need removal depending on local agricultural guidance.",
        "prevention": "Monitor and manage whiteflies and remove infected plant material."
    }
}


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==============================
# PREDICTION
# ==============================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Check image
    if "image" not in request.files:

        return "No image uploaded."


    file = request.files["image"]

    if file.filename == "":

        return "Please select an image."


    # Save image
    filename = secure_filename(
        file.filename
    )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(image_path)


    # Open image
    image = Image.open(
        image_path
    ).convert("RGB")


    # Transform image
    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)


    # AI prediction
    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    # Prediction result
    predicted_class = classes[
        prediction.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )


    # Get information
    info = disease_info.get(
        predicted_class,

        {
            "name": predicted_class,
            "scientific_name": "Not available",
            "status": "Prediction",
            "severity": "Unknown",
            "cause": "Information not available.",
            "symptoms": "Please consult an agricultural expert.",
            "action": "Inspect the plant carefully.",
            "treatment": "Seek local agricultural guidance.",
            "prevention": "Maintain good crop hygiene."
        }
    )


    # Confidence level
    if confidence_value >= 70:

        confidence_level = "High"

    elif confidence_value >= 40:

        confidence_level = "Medium"

    else:

        confidence_level = "Low"


    # Image URL
    image_url = "/" + image_path.replace(
        "\\",
        "/"
    )


    # Result page
    return render_template(
        "result.html",

        disease_name=info["name"],

        scientific_name=info[
            "scientific_name"
        ],

        status=info["status"],

        severity=info["severity"],

        cause=info["cause"],

        symptoms=info["symptoms"],

        action=info["action"],

        treatment=info["treatment"],

        prevention=info["prevention"],

        confidence=round(
            confidence_value,
            2
        ),

        confidence_level=confidence_level,

        image_url=image_url
    )


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )