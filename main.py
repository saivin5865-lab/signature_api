from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import io, json, joblib, os

app = FastAPI()

# Load your trained model and label encoder
clf = joblib.load("models/signature_verification_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Load reference data (JSON)
with open("data/reference_data.json", "r") as f:
    reference_data = json.load(f)

# Feature extractor (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return feature_model.predict(x).flatten()

@app.get("/")
def root():
    return {"message": "Signature comparison API is running!"}

@app.post("/compare")
async def compare_signature(name: str = Form(...), file: UploadFile = File(...)):
    if name not in reference_data:
        return JSONResponse({"error": f"No reference found for {name}"}, status_code=404)
    
    # Load reference image
    ref_path = reference_data[name]
    if not os.path.exists(ref_path):
        return JSONResponse({"error": f"Reference image not found: {ref_path}"}, status_code=404)
    
    with open(ref_path, "rb") as f:
        ref_bytes = f.read()

    # Extract features
    ref_feat = extract_features(ref_bytes)
    new_feat = extract_features(await file.read())

    # Compare using cosine similarity
    sim = cosine_similarity([ref_feat], [new_feat])[0][0]
    match = bool(sim >= 0.85)

    return JSONResponse({
        "name": name,
        "similarity": float(sim),
        "match": match
    })
