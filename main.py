from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io, json, joblib, os

app = FastAPI()

# --- Load models ---
MODEL_PATH = "models/signature_verification_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
JSON_PATH = "data/reference_data.json"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# --- Load JSON reference ---
with open(JSON_PATH, "r") as f:
    reference_data = json.load(f)

@app.post("/compare")
async def compare_signature(name: str = Form(...), file: UploadFile = None):
    if name not in reference_data:
        return JSONResponse({"error": "Name not found in reference data"}, status_code=404)

    ref_path = reference_data[name]
    if not os.path.exists(ref_path):
        return JSONResponse({"error": f"Reference image {ref_path} not found"}, status_code=404)

    # Load reference image
    ref_img = Image.open(ref_path).convert("L").resize((224, 224))
    ref_arr = np.array(ref_img).flatten().reshape(1, -1)

    # Load uploaded signature
    upload_img = Image.open(io.BytesIO(await file.read())).convert("L").resize((224, 224))
    upload_arr = np.array(upload_img).flatten().reshape(1, -1)

    # Compare using model (dummy example: similarity check)
    ref_pred = model.predict(ref_arr)
    upload_pred = model.predict(upload_arr)

    similarity = float(np.dot(ref_pred, upload_pred.T))
    result = similarity > 0.9  # adjust threshold

    return JSONResponse({
        "match": bool(result),
        "similarity": similarity,
        "name": name
    })
