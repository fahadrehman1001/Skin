from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import os

app = Flask(__name__)

# Load class names
class_names = [
    'Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases',
    'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles',
    'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites',
    'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors',
    'Vasculitis Photos', 'Warts Molluscum and other Viral Infections'
]

# Causes and Precautions Dictionary
info_dict = {
    "Acne and Rosacea Photos": {
        "cause": "Hormonal changes, stress, and bacteria.",
        "precaution": "Wash face twice daily, avoid oily foods, and use non-comedogenic skincare."
    },
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
        "cause": "Prolonged sun exposure and genetic factors.",
        "precaution": "Use sunscreen, wear protective clothing, and avoid excessive sun exposure."
    },
    "Atopic Dermatitis Photos": {
        "cause": "Genetic predisposition, immune system dysfunction, and environmental triggers.",
        "precaution": "Moisturize regularly, avoid allergens, and use mild soaps."
    },
    "Bullous Disease Photos": {
        "cause": "Autoimmune disorders causing skin blistering.",
        "precaution": "Avoid trauma to the skin, maintain hygiene, and follow prescribed medications."
    },
    "Cellulitis Impetigo and other Bacterial Infections": {
        "cause": "Bacterial infections from cuts, bites, or wounds.",
        "precaution": "Keep skin clean, treat wounds promptly, and avoid scratching."
    },
    "Eczema Photos": {
        "cause": "Genetics, allergens, and dry skin.",
        "precaution": "Moisturize daily, avoid triggers, and use mild, fragrance-free soaps."
    },
    "Exanthems and Drug Eruptions": {
        "cause": "Allergic reactions to medications or viral infections.",
        "precaution": "Identify and avoid allergenic drugs, consult a doctor for alternative treatments."
    },
    "Hair Loss Photos Alopecia and other Hair Diseases": {
        "cause": "Genetics, stress, and hormonal imbalances.",
        "precaution": "Maintain a healthy diet, avoid excessive hairstyling, and manage stress."
    },
    "Herpes HPV and other STDs Photos": {
        "cause": "Viral infections transmitted through contact.",
        "precaution": "Practice safe sex, maintain hygiene, and get vaccinated if possible."
    },
    "Light Diseases and Disorders of Pigmentation": {
        "cause": "Sun exposure, genetics, or autoimmune disorders.",
        "precaution": "Use sunscreen, wear protective clothing, and consult a dermatologist."
    },
    "Lupus and other Connective Tissue diseases": {
        "cause": "Autoimmune disorders causing systemic inflammation.",
        "precaution": "Avoid sun exposure, manage stress, and follow prescribed medications."
    },
    "Melanoma Skin Cancer Nevi and Moles": {
        "cause": "UV exposure and genetic factors.",
        "precaution": "Use sunscreen, wear protective clothing, and avoid tanning beds."
    },
    "Nail Fungus and other Nail Disease": {
        "cause": "Fungal infections, injury, or poor hygiene.",
        "precaution": "Keep nails dry and clean, avoid sharing nail tools, and use antifungal treatments."
    },
    "Poison Ivy Photos and other Contact Dermatitis": {
        "cause": "Allergic reactions to plants, chemicals, or irritants.",
        "precaution": "Avoid known irritants, wash skin immediately after exposure, and use anti-itch creams."
    },
    "Psoriasis pictures Lichen Planus and related diseases": {
        "cause": "Immune system dysfunction causing skin inflammation.",
        "precaution": "Moisturize, manage stress, and follow a prescribed treatment plan."
    },
    "Scabies Lyme Disease and other Infestations and Bites": {
        "cause": "Parasitic infections from mites, ticks, or insects.",
        "precaution": "Maintain personal hygiene, avoid close contact with infected individuals, and treat promptly."
    },
    "Seborrheic Keratoses and other Benign Tumors": {
        "cause": "Aging, genetics, and sun exposure.",
        "precaution": "Monitor skin changes, use sunscreen, and seek medical evaluation for growths."
    },
    "Systemic Disease": {
        "cause": "Underlying conditions affecting multiple organs.",
        "precaution": "Manage overall health, follow medical advice, and maintain a balanced lifestyle."
    },
    "Tinea Ringworm Candidiasis and other Fungal Infections": {
        "cause": "Fungal infections due to moisture, poor hygiene, or weak immunity.",
        "precaution": "Keep skin dry, wear breathable clothing, and use antifungal medications."
    },
    "Urticaria Hives": {
        "cause": "Allergic reactions, stress, or infections.",
        "precaution": "Avoid known allergens, use antihistamines, and manage stress."
    },
    "Vascular Tumors": {
        "cause": "Abnormal growth of blood vessels due to genetic factors.",
        "precaution": "Monitor growths, seek medical evaluation, and follow prescribed treatments."
    },
    "Vasculitis Photos": {
        "cause": "Inflammation of blood vessels due to autoimmune disorders or infections.",
        "precaution": "Follow a healthy diet, avoid infections, and take prescribed medications."
    },
    "Warts Molluscum and other Viral Infections": {
        "cause": "Viral infections causing skin growths.",
        "precaution": "Avoid direct skin contact, maintain hygiene, and use antiviral treatments."
    }
}


# Load Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load Model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(class_names)
)
model.load_state_dict(torch.load("vit_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image Preprocessing
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    
    return transform(image).unsqueeze(0)

# Prediction Function
def predict_image(image_path):
    image = process_image(image_path)
    
    with torch.no_grad():
        outputs = model(image).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100  # Convert to percentage

    cause = info_dict.get(predicted_label, {}).get("cause", "Unknown")
    precaution = info_dict.get(predicted_label, {}).get("precaution", "No specific precaution available.")

    return predicted_label, confidence_score, cause, precaution

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        file_path = os.path.join("static", file.filename)
        file.save(file_path)

        predicted_label, confidence_score, cause, precaution = predict_image(file_path)

        return render_template(
            "index.html", uploaded_image=file_path, 
            predicted_label=predicted_label, confidence_score=confidence_score,
            cause=cause, precaution=precaution
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
