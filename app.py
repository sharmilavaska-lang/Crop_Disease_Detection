import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from openai import OpenAI


# =========================================================
# FLASK SETUP
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# =========================================================
# AI API
# =========================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

AI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "crop_disease_model.pth"
)

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


# =========================================================
# IMAGE TRANSFORM
# =========================================================

IMAGE_SIZE = 128

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================================================
# DISEASE INFORMATION
# =========================================================

disease_info = {

    "Tomato___Bacterial_spot": {
        "disease_name": "Bacterial Spot",
        "disease_name_te": "బాక్టీరియల్ స్పాట్",

        "scientific_name": "Bacterial infection",
        "scientific_name_te": "బాక్టీరియా వల్ల కలిగే వ్యాధి",

        "status": "Disease Detected",
        "status_te": "వ్యాధి గుర్తించబడింది",

        "severity": "Moderate",
        "severity_te": "మధ్యస్థ తీవ్రత",

        "cause": "Bacterial infection, wet leaves and humid conditions.",
        "cause_te": "బాక్టీరియా, ఆకులు ఎక్కువసేపు తడిగా ఉండటం మరియు అధిక తేమ కారణంగా వస్తుంది.",

        "symptoms": "Small dark spots on leaves, stems and fruits. Yellow halos may appear around spots.",
        "symptoms_te": "ఆకులు, కాండం మరియు పండ్లపై చిన్న నల్లటి మచ్చలు కనిపిస్తాయి. మచ్చల చుట్టూ పసుపు వలయం ఉండవచ్చు.",

        "action": "Remove severely infected leaves and avoid overhead watering.",
        "action_te": "తీవ్రంగా సోకిన ఆకులను తొలగించి, పై నుంచి నీరు పోయడం నివారించండి.",

        "treatment": "Use an appropriate copper-based bactericide according to the product label and local agricultural guidance.",
        "treatment_te": "ఉత్పత్తి లేబుల్ మరియు స్థానిక వ్యవసాయ నిపుణుల సూచన ప్రకారం తగిన కాపర్ ఆధారిత బాక్టీరియాసైడ్ ఉపయోగించండి.",

        "prevention": "Use clean seeds, maintain spacing and avoid prolonged leaf wetness.",
        "prevention_te": "శుభ్రమైన విత్తనాలు ఉపయోగించండి, మొక్కల మధ్య తగిన దూరం ఉంచండి మరియు ఆకులు ఎక్కువసేపు తడిగా ఉండకుండా చూడండి."
    },

    "Tomato___Early_blight": {
        "disease_name": "Early Blight",
        "disease_name_te": "ఎర్లీ బ్లైట్",

        "scientific_name": "Alternaria solani",
        "scientific_name_te": "ఆల్టర్నేరియా సోలాని అనే శిలీంధ్రం",

        "status": "Disease Detected",
        "status_te": "వ్యాధి గుర్తించబడింది",

        "severity": "Moderate to High",
        "severity_te": "మధ్యస్థం నుండి అధిక తీవ్రత",

        "cause": "Fungal infection favored by warm, humid conditions.",
        "cause_te": "వెచ్చని మరియు తేమతో కూడిన వాతావరణంలో శిలీంధ్రం వల్ల వస్తుంది.",

        "symptoms": "Brown circular spots with concentric rings, usually on older leaves first.",
        "symptoms_te": "ముఖ్యంగా పాత ఆకులపై గుండ్రటి గోధుమ రంగు మచ్చలు, వాటిలో వలయాల మాదిరి గుర్తులు కనిపిస్తాయి.",

        "action": "Remove affected lower leaves and improve air circulation.",
        "action_te": "సోకిన దిగువ ఆకులను తొలగించి, మొక్కల మధ్య గాలి ప్రసరణ మెరుగుపరచండి.",

        "treatment": "Use a suitable fungicide according to the product label and local agricultural advice.",
        "treatment_te": "ఉత్పత్తి లేబుల్ మరియు స్థానిక వ్యవసాయ సూచనల ప్రకారం తగిన శిలీంద్రనాశిని ఉపయోగించండి.",

        "prevention": "Avoid overhead irrigation, rotate crops and remove infected plant debris.",
        "prevention_te": "పై నుంచి నీరు పోయడం నివారించండి, పంట మార్పిడి చేయండి మరియు సోకిన మొక్కల అవశేషాలను తొలగించండి."
    },

    "Tomato___healthy": {
        "disease_name": "Healthy Tomato",
        "disease_name_te": "ఆరోగ్యకరమైన టమాటా",

        "scientific_name": "No visible disease",
        "scientific_name_te": "కనిపించే వ్యాధి లక్షణాలు లేవు",

        "status": "Healthy",
        "status_te": "ఆరోగ్యంగా ఉంది",

        "severity": "None",
        "severity_te": "తీవ్రత లేదు",

        "cause": "No obvious disease detected.",
        "cause_te": "స్పష్టమైన వ్యాధి లక్షణాలు గుర్తించబడలేదు.",

        "symptoms": "The uploaded leaf appears healthy.",
        "symptoms_te": "మీరు upload చేసిన ఆకు ఆరోగ్యంగా కనిపిస్తోంది.",

        "action": "Continue regular crop care.",
        "action_te": "సాధారణ పంట సంరక్షణ కొనసాగించండి.",

        "treatment": "No disease treatment is required based on this prediction.",
        "treatment_te": "ఈ అంచనా ప్రకారం వ్యాధికి ప్రత్యేక చికిత్స అవసరం లేదు.",

        "prevention": "Maintain balanced watering, nutrition, sunlight and good field hygiene.",
        "prevention_te": "సమతుల్య నీరు, పోషకాలు, సూర్యకాంతి మరియు మంచి పొల పరిశుభ్రతను కొనసాగించండి."
    },

    "Tomato___Late_blight": {
        "disease_name": "Late Blight",
        "disease_name_te": "లేట్ బ్లైట్",

        "scientific_name": "Phytophthora infestans",
        "scientific_name_te": "ఫైటోఫ్తోరా ఇన్ఫెస్టాన్స్",

        "status": "Disease Detected",
        "status_te": "వ్యాధి గుర్తించబడింది",

        "severity": "High",
        "severity_te": "అధిక తీవ్రత",

        "cause": "A pathogen favored by cool, wet and humid conditions.",
        "cause_te": "చల్లని, తడి మరియు అధిక తేమ పరిస్థితుల్లో ఈ వ్యాధి వేగంగా వ్యాపిస్తుంది.",

        "symptoms": "Dark water-soaked lesions on leaves and stems; fruit may develop brown patches.",
        "symptoms_te": "ఆకులు మరియు కాండంపై నీటితో తడిసినట్లుగా నల్లటి మచ్చలు కనిపిస్తాయి; పండ్లపై గోధుమ మచ్చలు రావచ్చు.",

        "action": "Remove severely infected plant material and reduce leaf wetness.",
        "action_te": "తీవ్రంగా సోకిన మొక్క భాగాలను తొలగించి, ఆకులు తడిగా ఉండకుండా చూడండి.",

        "treatment": "Use a locally approved fungicide according to the label and agricultural guidance.",
        "treatment_te": "స్థానికంగా అనుమతించబడిన శిలీంద్రనాశిని లేబుల్ మరియు వ్యవసాయ నిపుణుల సూచన ప్రకారం ఉపయోగించండి.",

        "prevention": "Improve air circulation and avoid overhead watering.",
        "prevention_te": "గాలి ప్రసరణ మెరుగుపరచండి మరియు పై నుంచి నీరు పోయడం నివారించండి."
    },

    "Tomato___Leaf_Mold": {
        "disease_name": "Leaf Mold",
        "disease_name_te": "లీఫ్ మోల్డ్",

        "scientific_name": "Passalora fulva",
        "scientific_name_te": "పస్సలోరా ఫుల్వా",

        "status": "Disease Detected",
        "status_te": "వ్యాధి గుర్తించబడింది",

        "severity": "Moderate",
        "severity_te": "మధ్యస్థ తీవ్రత",

        "cause": "Fungal disease favored by high humidity and poor air circulation.",
        "cause_te": "అధిక తేమ మరియు తక్కువ గాలి ప్రసరణ కారణంగా శిలీంధ్రం అభివృద్ధి చెందుతుంది.",

        "symptoms": "Yellow patches on the upper leaf surface and olive-green to brown growth underneath.",
        "symptoms_te": "ఆకు పైభాగంలో పసుపు మచ్చలు మరియు దిగువ భాగంలో ఆలివ్ ఆకుపచ్చ లేదా గోధుమ రంగు పొర కనిపిస్తుంది.",

        "action": "Remove infected leaves and improve ventilation.",
        "action_te": "సోకిన ఆకులను తొలగించి, గాలి ప్రసరణ మెరుగుపరచండి.",

        "treatment": "Use a suitable fungicide according to local recommendations.",
        "treatment_te": "స్థానిక వ్యవసాయ సూచనల ప్రకారం తగిన శిలీంద్రనాశిని ఉపయోగించండి.",

        "prevention": "Avoid excessive humidity and provide adequate plant spacing.",
        "prevention_te": "అధిక తేమను నివారించి, మొక్కల మధ్య తగిన దూరం ఉంచండి."
    },

    "Tomato___Septoria_leaf_spot": {
        "disease_name": "Septoria Leaf Spot",
        "disease_name_te": "సెప్టోరియా లీఫ్ స్పాట్",

        "scientific_name": "Septoria lycopersici",
        "scientific_name_te": "సెప్టోరియా లైకోపెర్సిసి",

        "status": "Disease Detected",
        "status_te": "వ్యాధి గుర్తించబడింది",

        "severity": "Moderate",
        "severity_te": "మధ్యస్థ తీవ్రత",

        "cause": "Fungal infection spread by moisture and infected plant debris.",
        "cause_te": "తేమ మరియు సోకిన మొక్కల అవశేషాల ద్వారా వ్యాపించే శిలీంధ్ర వ్యాధి.",

        "symptoms": "Small circular spots with dark margins and pale centers.",
        "symptoms_te": "ముదురు అంచులు మరియు లేత మధ్యభాగంతో చిన్న గుండ్రటి మచ్చలు కనిపిస్తాయి.",

        "action": "Remove infected leaves and keep foliage dry.",
        "action_te": "సోకిన ఆకులను తొలగించి, ఆకులు పొడిగా ఉండేలా చూడండి.",

        "treatment": "Use a suitable fungicide following its label.",
        "treatment_te": "లేబుల్ సూచనల ప్రకారం తగిన శిలీంద్రనాశిని ఉపయోగించండి.",

        "prevention": "Remove crop debris and avoid overhead irrigation.",
        "prevention_te": "పంట అవశేషాలను తొలగించి, పై నుంచి నీరు పోయడం నివారించండి."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease_name": "Two-Spotted Spider Mite",
        "disease_name_te": "టూ-స్పాటెడ్ స్పైడర్ మైట్",

        "scientific_name": "Tetranychus urticae",
        "scientific_name_te": "టెట్రానైకస్ ఉర్టికే",

        "status": "Pest Detected",
        "status_te": "పురుగు సమస్య గుర్తించబడింది",

        "severity": "Moderate",
        "severity_te": "మధ్యస్థ తీవ్రత",

        "cause": "Spider mite infestation, often favored by hot and dry conditions.",
        "cause_te": "స్పైడర్ మైట్ పురుగు సోకడం; వేడి మరియు పొడి వాతావరణంలో ఎక్కువగా కనిపిస్తుంది.",

        "symptoms": "Tiny speckles, yellowing leaves and fine webbing may appear.",
        "symptoms_te": "చిన్న చిన్న మచ్చలు, ఆకులు పసుపు రంగులోకి మారడం మరియు సన్నని జాలం కనిపించవచ్చు.",

        "action": "Inspect leaf undersides and remove heavily affected leaves.",
        "action_te": "ఆకుల దిగువ భాగాన్ని పరిశీలించి, తీవ్రంగా సోకిన ఆకులను తొలగించండి.",

        "treatment": "Use an appropriate registered miticide or biological control according to local guidance.",
        "treatment_te": "స్థానిక వ్యవసాయ సూచనల ప్రకారం అనుమతించబడిన మైటిసైడ్ లేదా జీవ నియంత్రణ పద్ధతిని ఉపయోగించండి.",

        "prevention": "Reduce plant stress and maintain suitable field conditions.",
        "prevention_te": "మొక్కలపై ఒత్తిడి తగ్గించి, తగిన పంట పరిస్థితులను నిర్వహించండి."
    },

    "Tomato___Target_Spot": {
        "disease_name": "Target Spot",
        "disease_name_te": "టార్గెట్ స్పాట్",

        "scientific_name": "Corynespora cassiicola",
        "scientific_name_te": "కోరినెస్పోరా కాసికోలా",

        "status": "Disease Detected",
        "status_te": "వ్యాధి గుర్తించబడింది",

        "severity": "Moderate",
        "severity_te": "మధ్యస్థ తీవ్రత",

        "cause": "Fungal infection favored by warm and humid conditions.",
        "cause_te": "వెచ్చని మరియు తేమతో కూడిన వాతావరణంలో శిలీంధ్రం వల్ల వస్తుంది.",

        "symptoms": "Circular brown lesions with concentric target-like rings.",
        "symptoms_te": "టార్గెట్ లాంటి వలయాలతో గుండ్రటి గోధుమ రంగు మచ్చలు కనిపిస్తాయి.",

        "action": "Remove severely affected leaves and improve airflow.",
        "action_te": "తీవ్రంగా సోకిన ఆకులను తొలగించి, గాలి ప్రసరణ మెరుగుపరచండి.",

        "treatment": "Apply an appropriate fungicide according to local recommendations.",
        "treatment_te": "స్థానిక వ్యవసాయ సూచనల ప్రకారం తగిన శిలీంద్రనాశిని ఉపయోగించండి.",

        "prevention": "Avoid prolonged leaf wetness and maintain field sanitation.",
        "prevention_te": "ఆకులు ఎక్కువసేపు తడిగా ఉండకుండా చూసి, పొల పరిశుభ్రత పాటించండి."
    },

    "Tomato___Tomato_mosaic_virus": {
        "disease_name": "Tomato Mosaic Virus",
        "disease_name_te": "టమాటా మోసాయిక్ వైరస్",

        "scientific_name": "Tomato mosaic virus",
        "scientific_name_te": "టమాటా మోసాయిక్ వైరస్",

        "status": "Viral Disease Detected",
        "status_te": "వైరస్ వ్యాధి గుర్తించబడింది",

        "severity": "High",
        "severity_te": "అధిక తీవ్రత",

        "cause": "Viral infection that can spread through contaminated hands, tools and plant material.",
        "cause_te": "కలుషితమైన చేతులు, పనిముట్లు మరియు మొక్కల పదార్థాల ద్వారా వ్యాపించే వైరస్.",

        "symptoms": "Mosaic patterns, mottled leaves and possible plant growth reduction.",
        "symptoms_te": "ఆకులపై మోసాయిక్ ఆకృతులు, రంగు మార్పులు మరియు మొక్క పెరుగుదల తగ్గడం కనిపించవచ్చు.",

        "action": "Remove infected plants and disinfect tools.",
        "action_te": "సోకిన మొక్కలను తొలగించి, పనిముట్లను శుభ్రపరచండి.",

        "treatment": "There is no curative treatment for an infected plant; focus on sanitation and removal.",
        "treatment_te": "సోకిన మొక్కకు వైరస్‌ను పూర్తిగా నయం చేసే చికిత్స లేదు; పరిశుభ్రత మరియు సోకిన మొక్కల తొలగింపుపై దృష్టి పెట్టండి.",

        "prevention": "Use clean planting material and sanitize tools and hands.",
        "prevention_te": "శుభ్రమైన నాటే పదార్థం ఉపయోగించి, చేతులు మరియు పనిముట్లను శుభ్రంగా ఉంచండి."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease_name": "Tomato Yellow Leaf Curl Virus",
        "disease_name_te": "టమాటా ఎల్లో లీఫ్ కర్ల్ వైరస్",

        "scientific_name": "Tomato yellow leaf curl virus",
        "scientific_name_te": "టమాటా ఎల్లో లీఫ్ కర్ల్ వైరస్",

        "status": "Viral Disease Detected",
        "status_te": "వైరస్ వ్యాధి గుర్తించబడింది",

        "severity": "High",
        "severity_te": "అధిక తీవ్రత",

        "cause": "Virus commonly transmitted by whiteflies.",
        "cause_te": "వైట్‌ఫ్లై పురుగుల ద్వారా వైరస్ సాధారణంగా వ్యాపిస్తుంది.",

        "symptoms": "Yellowing, upward leaf curling, stunted growth and reduced fruit production.",
        "symptoms_te": "ఆకులు పసుపు రంగులోకి మారడం, పైకి ముడుచుకోవడం, మొక్క పెరుగుదల తగ్గడం మరియు పండ్ల ఉత్పత్తి తగ్గడం.",

        "action": "Remove severely infected plants and manage whiteflies.",
        "action_te": "తీవ్రంగా సోకిన మొక్కలను తొలగించి, వైట్‌ఫ్లై పురుగులను నియంత్రించండి.",

        "treatment": "There is no direct cure for the virus; control the vector and remove infected plants.",
        "treatment_te": "వైరస్‌కు నేరుగా నివారణ చికిత్స లేదు; వైరస్‌ను వ్యాప్తి చేసే పురుగులను నియంత్రించి, సోకిన మొక్కలను తొలగించండి.",

        "prevention": "Use healthy seedlings and monitor whitefly populations.",
        "prevention_te": "ఆరోగ్యకరమైన మొక్కల నాట్లను ఉపయోగించి, వైట్‌ఫ్లై పురుగుల సంఖ్యను పర్యవేక్షించండి."
    }
}


# =========================================================
# HELPER
# =========================================================

def get_confidence_level(confidence):

    if confidence >= 70:
        return "High", "అధిక నమ్మక స్థాయి"

    elif confidence >= 40:
        return "Medium", "మధ్యస్థ నమ్మక స్థాయి"

    return "Low", "తక్కువ నమ్మక స్థాయి"


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":
        return "No file selected", 400

    allowed = {"jpg", "jpeg", "png"}

    extension = file.filename.rsplit(".", 1)[-1].lower()

    if extension not in allowed:
        return "Only JPG, JPEG and PNG images are allowed.", 400

    filename = (
        str(uuid.uuid4())
        + "_"
        + secure_filename(file.filename)
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    try:

        image = Image.open(filepath).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        predicted_class = classes[predicted.item()]

        confidence_value = float(
            confidence.item() * 100
        )

        confidence_level, confidence_level_te = \
            get_confidence_level(confidence_value)

        info = disease_info.get(
            predicted_class,
            disease_info["Tomato___healthy"]
        )

        image_url = "/static/uploads/" + filename

        return render_template(
            "result.html",

            crop_name="Tomato",
            crop_name_te="టమాటా",

            disease_name=info["disease_name"],
            disease_name_te=info["disease_name_te"],

            scientific_name=info["scientific_name"],
            scientific_name_te=info["scientific_name_te"],

            status=info["status"],
            status_te=info["status_te"],

            severity=info["severity"],
            severity_te=info["severity_te"],

            cause=info["cause"],
            cause_te=info["cause_te"],

            symptoms=info["symptoms"],
            symptoms_te=info["symptoms_te"],

            action=info["action"],
            action_te=info["action_te"],

            treatment=info["treatment"],
            treatment_te=info["treatment_te"],

            prevention=info["prevention"],
            prevention_te=info["prevention_te"],

            confidence=round(confidence_value, 2),
            confidence_level=confidence_level,
            confidence_level_te=confidence_level_te,

            image_url=image_url
        )

    except Exception as e:

        print("Prediction error:", e)

        return "Prediction failed. Please try another image.", 500


# =========================================================
# AI QUESTION ANSWER
# =========================================================

@app.route("/ask-ai", methods=["POST"])
def ask_ai():

    data = request.get_json(silent=True) or {}

    question = str(
        data.get("question", "")
    ).strip()

    language = data.get(
        "language",
        "en"
    )

    result_context = data.get(
        "result_context",
        {}
    )

    if not question:

        return jsonify({
            "success": False,
            "answer": (
                "Please enter a question."
                if language == "en"
                else "దయచేసి ఒక ప్రశ్న అడగండి."
            )
        }), 400

    if len(question) > 2000:

        return jsonify({
            "success": False,
            "answer": (
                "Please keep your question shorter."
                if language == "en"
                else "దయచేసి ప్రశ్నను కొంచెం చిన్నగా అడగండి."
            )
        }), 400

    # -----------------------------------------------------
    # API KEY CHECK
    # -----------------------------------------------------

    if client is None:

        return jsonify({
            "success": False,
            "answer": (
                "AI service is not connected yet. "
                "Please add OPENAI_API_KEY to your environment variables."
                if language == "en"
                else
                "AI సేవ ఇంకా కనెక్ట్ కాలేదు. "
                "OPENAI_API_KEY ను environment variables లో add చేయండి."
            )
        })

    # -----------------------------------------------------
    # LANGUAGE
    # -----------------------------------------------------

    if language == "te":

        language_instruction = """
You MUST answer in Telugu.
Use simple, natural Telugu that farmers can easily understand.
Do not unnecessarily use difficult scientific terminology.
If a scientific term is necessary, write the English term in brackets.
"""

    else:

        language_instruction = """
Answer in clear, simple English.
Use practical farming language.
"""

    # -----------------------------------------------------
    # RESULT CONTEXT
    # -----------------------------------------------------

    context_text = ""

    if result_context:

        context_text = f"""
The user has an AI crop-detection result.

Crop: {result_context.get("crop", "Tomato")}
Disease: {result_context.get("disease", "")}
Scientific name: {result_context.get("scientific_name", "")}
Status: {result_context.get("status", "")}
Severity: {result_context.get("severity", "")}
Confidence: {result_context.get("confidence", "")}%
Cause: {result_context.get("cause", "")}
Symptoms: {result_context.get("symptoms", "")}
Recommended action: {result_context.get("action", "")}
Treatment: {result_context.get("treatment", "")}
Prevention: {result_context.get("prevention", "")}
"""

    # -----------------------------------------------------
    # SYSTEM INSTRUCTION
    # -----------------------------------------------------

    instructions = f"""
You are AI Crop Doctor, an agricultural assistant for farmers.

{language_instruction}

{context_text}

Answer the farmer's question directly.

You can help with:
- crop diseases
- plant symptoms
- pests
- watering
- fertilizers
- soil
- crop care
- disease prevention
- general tomato cultivation
- interpreting the current AI detection result

Important:
1. Do not claim that an image-based prediction is 100% certain.
2. If the question is about pesticides, fungicides, insecticides or chemicals,
   recommend following the product label and local agricultural guidance.
3. Do not invent exact chemical doses when the product formulation is unknown.
4. Give practical steps whenever possible.
5. Keep answers concise but useful.
6. If the user asks something unrelated to farming, politely say that you are
   specialized in crop and agricultural questions.
"""

    try:

        response = client.responses.create(
            model=AI_MODEL,
            instructions=instructions,
            input=question
        )

        answer = response.output_text.strip()

        if not answer:

            answer = (
                "I could not generate an answer."
                if language == "en"
                else "సమాధానం రూపొందించలేకపోయాను."
            )

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as e:

        print("AI error:", e)

        return jsonify({
            "success": False,
            "answer": (
                "AI service is temporarily unavailable. Please try again."
                if language == "en"
                else
                "AI సేవ ప్రస్తుతం అందుబాటులో లేదు. కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి."
            )
        }), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )