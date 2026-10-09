
import os
import uuid
import logging

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torchvision import models, transforms
from openai import OpenAI


# =========================================================
# LOGGING AND FLASK SETUP
# =========================================================

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# CPU మీద inference కోసం threads పరిమితం చేయడం
torch.set_num_threads(1)


# =========================================================
# OPENAI API
# =========================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
AI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")


# =========================================================
# LOAD CROP DISEASE MODEL
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "crop_disease_model.pth"
)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

app.logger.info("Loading crop disease model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

app.logger.info("Crop disease model loaded successfully.")


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

def make_info(
    name, name_te, scientific, scientific_te,
    severity, severity_te, cause, cause_te,
    symptoms, symptoms_te, action, action_te,
    treatment, treatment_te, prevention, prevention_te,
    status="Disease Detected", status_te="వ్యాధి గుర్తించబడింది"
):
    return {
        "disease_name": name,
        "disease_name_te": name_te,
        "scientific_name": scientific,
        "scientific_name_te": scientific_te,
        "status": status,
        "status_te": status_te,
        "severity": severity,
        "severity_te": severity_te,
        "cause": cause,
        "cause_te": cause_te,
        "symptoms": symptoms,
        "symptoms_te": symptoms_te,
        "action": action,
        "action_te": action_te,
        "treatment": treatment,
        "treatment_te": treatment_te,
        "prevention": prevention,
        "prevention_te": prevention_te
    }


disease_info = {
    "Tomato___Bacterial_spot": make_info(
        "Bacterial Spot", "బాక్టీరియల్ స్పాట్",
        "Bacterial infection", "బాక్టీరియా వల్ల కలిగే వ్యాధి",
        "Moderate", "మధ్యస్థ తీవ్రత",
        "Bacterial infection, wet leaves and humid conditions.",
        "బాక్టీరియా, ఆకులు ఎక్కువసేపు తడిగా ఉండటం మరియు అధిక తేమ కారణంగా వస్తుంది.",
        "Small dark spots on leaves, stems and fruits.",
        "ఆకులు, కాండం మరియు పండ్లపై చిన్న నల్లటి మచ్చలు కనిపిస్తాయి.",
        "Remove severely infected leaves and avoid overhead watering.",
        "తీవ్రంగా సోకిన ఆకులను తొలగించి, పై నుంచి నీరు పోయడం నివారించండి.",
        "Use an appropriate copper-based bactericide according to the label and local guidance.",
        "లేబుల్ మరియు స్థానిక వ్యవసాయ నిపుణుల సూచన ప్రకారం తగిన కాపర్ ఆధారిత మందు ఉపయోగించండి.",
        "Use clean seeds and avoid prolonged leaf wetness.",
        "శుభ్రమైన విత్తనాలు ఉపయోగించి ఆకులు ఎక్కువసేపు తడిగా ఉండకుండా చూడండి."
    ),

    "Tomato___Early_blight": make_info(
        "Early Blight", "ఎర్లీ బ్లైట్",
        "Alternaria solani", "ఆల్టర్నేరియా సోలాని అనే శిలీంధ్రం",
        "Moderate to High", "మధ్యస్థం నుండి అధిక తీవ్రత",
        "Fungal infection favored by warm, humid conditions.",
        "వెచ్చని మరియు తేమతో కూడిన వాతావరణంలో శిలీంధ్రం వల్ల వస్తుంది.",
        "Brown circular spots with concentric rings, often on older leaves.",
        "ముఖ్యంగా పాత ఆకులపై వలయాల మాదిరి గోధుమ మచ్చలు కనిపిస్తాయి.",
        "Remove affected lower leaves and improve air circulation.",
        "సోకిన దిగువ ఆకులను తొలగించి గాలి ప్రసరణ మెరుగుపరచండి.",
        "Use a suitable fungicide according to the label and local advice.",
        "లేబుల్ మరియు స్థానిక వ్యవసాయ సూచనల ప్రకారం తగిన శిలీంద్రనాశిని ఉపయోగించండి.",
        "Rotate crops and remove infected plant debris.",
        "పంట మార్పిడి చేసి, సోకిన మొక్కల అవశేషాలను తొలగించండి."
    ),

    "Tomato___healthy": make_info(
        "Healthy Tomato", "ఆరోగ్యకరమైన టమాటా",
        "No visible disease", "కనిపించే వ్యాధి లక్షణాలు లేవు",
        "None", "తీవ్రత లేదు",
        "No obvious disease detected.",
        "స్పష్టమైన వ్యాధి లక్షణాలు గుర్తించబడలేదు.",
        "The uploaded leaf appears healthy.",
        "మీరు upload చేసిన ఆకు ఆరోగ్యంగా కనిపిస్తోంది.",
        "Continue regular crop care.",
        "సాధారణ పంట సంరక్షణ కొనసాగించండి.",
        "No disease treatment is required based on this prediction.",
        "ఈ అంచనా ప్రకారం ప్రత్యేక వ్యాధి చికిత్స అవసరం లేదు.",
        "Maintain balanced watering, nutrition and field hygiene.",
        "సమతుల్య నీరు, పోషకాలు మరియు పొల పరిశుభ్రతను కొనసాగించండి.",
        status="Healthy", status_te="ఆరోగ్యంగా ఉంది"
    ),

    "Tomato___Late_blight": make_info(
        "Late Blight", "లేట్ బ్లైట్",
        "Phytophthora infestans", "ఫైటోఫ్తోరా ఇన్ఫెస్టాన్స్",
        "High", "అధిక తీవ్రత",
        "A pathogen favored by cool, wet and humid conditions.",
        "చల్లని, తడి మరియు అధిక తేమ పరిస్థితుల్లో ఈ వ్యాధి వేగంగా వ్యాపిస్తుంది.",
        "Dark water-soaked lesions on leaves and stems.",
        "ఆకులు మరియు కాండంపై నీటితో తడిసినట్లుగా నల్లటి మచ్చలు కనిపిస్తాయి.",
        "Remove severely infected plant material and reduce leaf wetness.",
        "తీవ్రంగా సోకిన మొక్క భాగాలను తొలగించి ఆకులు తడిగా ఉండకుండా చూడండి.",
        "Use a locally approved fungicide according to its label.",
        "స్థానికంగా అనుమతించబడిన శిలీంద్రనాశిని లేబుల్ ప్రకారం ఉపయోగించండి.",
        "Improve air circulation and avoid overhead watering.",
        "గాలి ప్రసరణ మెరుగుపరచి పై నుంచి నీరు పోయడం నివారించండి."
    ),

    "Tomato___Leaf_Mold": make_info(
        "Leaf Mold", "లీఫ్ మోల్డ్",
        "Passalora fulva", "పస్సలోరా ఫుల్వా",
        "Moderate", "మధ్యస్థ తీవ్రత",
        "Fungal disease favored by high humidity.",
        "అధిక తేమ వల్ల శిలీంధ్ర వ్యాధి అభివృద్ధి చెందుతుంది.",
        "Yellow patches above leaves and olive-brown growth underneath.",
        "ఆకు పైభాగంలో పసుపు మచ్చలు, దిగువన ఆలివ్ లేదా గోధుమ రంగు పొర కనిపించవచ్చు.",
        "Remove infected leaves and improve ventilation.",
        "సోకిన ఆకులను తొలగించి గాలి ప్రసరణ మెరుగుపరచండి.",
        "Use a suitable fungicide according to local recommendations.",
        "స్థానిక వ్యవసాయ సూచనల ప్రకారం తగిన శిలీంద్రనాశిని ఉపయోగించండి.",
        "Avoid excessive humidity and provide plant spacing.",
        "అధిక తేమను నివారించి మొక్కల మధ్య తగిన దూరం ఉంచండి."
    ),

    "Tomato___Septoria_leaf_spot": make_info(
        "Septoria Leaf Spot", "సెప్టోరియా లీఫ్ స్పాట్",
        "Septoria lycopersici", "సెప్టోరియా లైకోపెర్సిసి",
        "Moderate", "మధ్యస్థ తీవ్రత",
        "Fungal infection spread by moisture and infected debris.",
        "తేమ మరియు సోకిన మొక్కల అవశేషాల ద్వారా వ్యాపించే శిలీంధ్ర వ్యాధి.",
        "Small circular spots with dark margins and pale centers.",
        "ముదురు అంచులు, లేత మధ్యభాగంతో చిన్న గుండ్రటి మచ్చలు కనిపిస్తాయి.",
        "Remove infected leaves and keep foliage dry.",
        "సోకిన ఆకులను తొలగించి ఆకులు పొడిగా ఉండేలా చూడండి.",
        "Use a suitable fungicide following its label.",
        "లేబుల్ సూచనల ప్రకారం తగిన శిలీంద్రనాశిని ఉపయోగించండి.",
        "Remove crop debris and avoid overhead irrigation.",
        "పంట అవశేషాలను తొలగించి పై నుంచి నీరు పోయడం నివారించండి."
    ),

    "Tomato___Spider_mites Two-spotted_spider_mite": make_info(
        "Two-Spotted Spider Mite", "టూ-స్పాటెడ్ స్పైడర్ మైట్",
        "Tetranychus urticae", "టెట్రానైకస్ ఉర్టికే",
        "Moderate", "మధ్యస్థ తీవ్రత",
        "Spider mites are often favored by hot, dry conditions.",
        "వేడి మరియు పొడి వాతావరణంలో స్పైడర్ మైట్లు ఎక్కువగా కనిపిస్తాయి.",
        "Tiny speckles, yellowing leaves and fine webbing.",
        "చిన్న మచ్చలు, ఆకులు పసుపు రంగులోకి మారడం, సన్నని జాలం కనిపించవచ్చు.",
        "Inspect leaf undersides and remove heavily affected leaves.",
        "ఆకుల దిగువ భాగాన్ని పరిశీలించి తీవ్రంగా సోకిన ఆకులను తొలగించండి.",
        "Use a registered treatment according to local guidance.",
        "స్థానిక వ్యవసాయ సూచనల ప్రకారం అనుమతించబడిన చికిత్సను ఉపయోగించండి.",
        "Reduce plant stress and monitor pests.",
        "మొక్కలపై ఒత్తిడిని తగ్గించి పురుగులను పర్యవేక్షించండి.",
        status="Pest Detected", status_te="పురుగు సమస్య గుర్తించబడింది"
    ),

    "Tomato___Target_Spot": make_info(
        "Target Spot", "టార్గెట్ స్పాట్",
        "Corynespora cassiicola", "కోరినెస్పోరా కాసికోలా",
        "Moderate", "మధ్యస్థ తీవ్రత",
        "Fungal infection favored by warm, humid conditions.",
        "వెచ్చని మరియు తేమతో కూడిన వాతావరణంలో శిలీంధ్రం వల్ల వస్తుంది.",
        "Circular brown lesions with target-like rings.",
        "టార్గెట్ లాంటి వలయాలతో గుండ్రటి గోధుమ మచ్చలు కనిపిస్తాయి.",
        "Remove affected leaves and improve airflow.",
        "సోకిన ఆకులను తొలగించి గాలి ప్రసరణ మెరుగుపరచండి.",
        "Apply an appropriate fungicide according to local recommendations.",
        "స్థానిక వ్యవసాయ సూచనల ప్రకారం తగిన శిలీంద్రనాశిని ఉపయోగించండి.",
        "Avoid prolonged leaf wetness and maintain field sanitation.",
        "ఆకులు ఎక్కువసేపు తడిగా ఉండకుండా చూసి పొల పరిశుభ్రత పాటించండి."
    ),

    "Tomato___Tomato_mosaic_virus": make_info(
        "Tomato Mosaic Virus", "టమాటా మోసాయిక్ వైరస్",
        "Tomato mosaic virus", "టమాటా మోసాయిక్ వైరస్",
        "High", "అధిక తీవ్రత",
        "Virus can spread through contaminated hands and tools.",
        "కలుషితమైన చేతులు మరియు పనిముట్ల ద్వారా వైరస్ వ్యాపించవచ్చు.",
        "Mosaic patterns, mottled leaves and reduced growth.",
        "ఆకులపై మోసాయిక్ ఆకృతులు, రంగు మార్పులు, పెరుగుదల తగ్గడం కనిపించవచ్చు.",
        "Remove infected plants and disinfect tools.",
        "సోకిన మొక్కలను తొలగించి పనిముట్లను శుభ్రపరచండి.",
        "There is no curative treatment for the infected plant.",
        "సోకిన మొక్కకు వైరస్‌ను పూర్తిగా నయం చేసే చికిత్స లేదు.",
        "Use clean planting material and sanitize tools.",
        "శుభ్రమైన నాటే పదార్థం ఉపయోగించి పనిముట్లను శుభ్రంగా ఉంచండి.",
        status="Viral Disease Detected", status_te="వైరస్ వ్యాధి గుర్తించబడింది"
    ),

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": make_info(
        "Tomato Yellow Leaf Curl Virus", "టమాటా ఎల్లో లీఫ్ కర్ల్ వైరస్",
        "Tomato yellow leaf curl virus", "టమాటా ఎల్లో లీఫ్ కర్ల్ వైరస్",
        "High", "అధిక తీవ్రత",
        "Virus commonly transmitted by whiteflies.",
        "వైట్‌ఫ్లై పురుగుల ద్వారా వైరస్ సాధారణంగా వ్యాపిస్తుంది.",
        "Yellowing, upward leaf curling and stunted growth.",
        "ఆకులు పసుపు రంగులోకి మారడం, పైకి ముడుచుకోవడం, పెరుగుదల తగ్గడం కనిపించవచ్చు.",
        "Remove severely infected plants and manage whiteflies.",
        "తీవ్రంగా సోకిన మొక్కలను తొలగించి వైట్‌ఫ్లై పురుగులను నియంత్రించండి.",
        "There is no direct cure; manage the vector and remove infected plants.",
        "వైరస్‌కు నేరుగా నివారణ లేదు; వైరస్ వ్యాప్తి చేసే పురుగులను నియంత్రించండి.",
        "Use healthy seedlings and monitor whiteflies.",
        "ఆరోగ్యకరమైన నాట్లను ఉపయోగించి వైట్‌ఫ్లై పురుగులను పర్యవేక్షించండి.",
        status="Viral Disease Detected", status_te="వైరస్ వ్యాధి గుర్తించబడింది"
    )
}


# =========================================================
# HELPERS
# =========================================================

def get_confidence_level(confidence):
    if confidence >= 70:
        return "High", "అధిక నమ్మక స్థాయి"
    elif confidence >= 40:
        return "Medium", "మధ్యస్థ నమ్మక స్థాయి"
    return "Low", "తక్కువ నమ్మక స్థాయి"


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# PREDICT DISEASE
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    app.logger.info("PREDICT ROUTE STARTED")

    try:
        if "file" not in request.files:
            app.logger.warning("No file in request")
            return "No file uploaded", 400

        file = request.files["file"]

        if not file.filename:
            app.logger.warning("No file selected")
            return "No file selected", 400

        allowed = {"jpg", "jpeg", "png"}
        if "." not in file.filename:
            return "Invalid filename", 400

        extension = file.filename.rsplit(".", 1)[-1].lower()
        if extension not in allowed:
            return "Only JPG, JPEG and PNG images are allowed.", 400

        filename = (
            str(uuid.uuid4()) + "_" + secure_filename(file.filename)
        )
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)
        app.logger.info("IMAGE SAVED: %s", filename)

        app.logger.info("Opening image...")
        image = Image.open(filepath).convert("RGB")

        app.logger.info("Preparing image tensor...")
        tensor = transform(image).unsqueeze(0).to(device)

        app.logger.info("MODEL PREDICTION STARTED")
        with torch.inference_mode():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        app.logger.info("MODEL PREDICTION FINISHED")

        predicted_class = classes[predicted.item()]
        confidence_value = float(confidence.item() * 100)

        confidence_level, confidence_level_te = (
            get_confidence_level(confidence_value)
        )

        info = disease_info.get(
            predicted_class,
            disease_info["Tomato___healthy"]
        )

        app.logger.info(
            "PREDICTION RESULT: %s, confidence=%.2f%%",
            predicted_class, confidence_value
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

    except UnidentifiedImageError:
        app.logger.exception("Uploaded file is not a valid image")
        return "Uploaded file is not a valid image.", 400

    except Exception:
        app.logger.exception("PREDICTION FAILED")
        return "Prediction failed. Check Render Logs for details.", 500


# =========================================================
# AI QUESTION ANSWER
# =========================================================

@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    language = data.get("language", "en")
    result_context = data.get("result_context", {})

    if not question:
        return jsonify({
            "success": False,
            "answer": (
                "దయచేసి ఒక ప్రశ్న అడగండి."
                if language == "te" else "Please enter a question."
            )
        }), 400

    if len(question) > 2000:
        return jsonify({
            "success": False,
            "answer": (
                "దయచేసి ప్రశ్నను కొంచెం చిన్నగా అడగండి."
                if language == "te"
                else "Please keep your question shorter."
            )
        }), 400

    if client is None:
        return jsonify({
            "success": False,
            "answer": (
                "AI సేవ ఇంకా కనెక్ట్ కాలేదు. OPENAI_API_KEY ను environment variables లో add చేయండి."
                if language == "te"
                else "AI service is not connected. Add OPENAI_API_KEY to environment variables."
            )
        }), 503

    if language == "te":
        language_instruction = (
            "Answer in simple, natural Telugu that farmers can understand. "
            "If a scientific term is necessary, include the English term in brackets."
        )
    else:
        language_instruction = "Answer in clear, simple English using practical farming language."

    context_text = ""
    if result_context:
        context_text = f"""
Current crop detection result:
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

    instructions = f"""
You are AI Crop Doctor, an agricultural assistant for farmers.
{language_instruction}
{context_text}

Answer directly and practically. Do not claim an image prediction is 100% certain.
For pesticides and other chemicals, follow the product label and local agricultural guidance.
Do not invent chemical doses. Keep answers concise and useful.
For unrelated questions, politely explain that you specialize in agriculture.
"""

    try:
        response = client.responses.create(
            model=AI_MODEL,
            instructions=instructions,
            input=question
        )

        answer = (response.output_text or "").strip()
        if not answer:
            answer = (
                "సమాధానం రూపొందించలేకపోయాను."
                if language == "te"
                else "I could not generate an answer."
            )

        return jsonify({"success": True, "answer": answer})

    except Exception:
        app.logger.exception("AI QUESTION FAILED")
        return jsonify({
            "success": False,
            "answer": (
                "AI సేవ ప్రస్తుతం అందుబాటులో లేదు. కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి."
                if language == "te"
                else "AI service is temporarily unavailable. Please try again."
            )
        }), 500


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
