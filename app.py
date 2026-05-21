# ==========================================
#  BACKEND INTERFACE SCHEMA DEFINITION
# ==========================================

import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Inference Engine Platform", 
    description="Production Capstone Toxicity Verification Portal V1"
)

class TextSubmission(BaseModel):
    text: str

model = None
vectorizer = None

@app.on_event("startup")
def boot_inference_libraries():
    """Triggered automatically as soon as the server boots up to map models into memory."""
    global model, vectorizer
    m_path = "models/complex_model.pkl"
    v_path = "models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(m_path) or not os.path.exists(v_path):
        raise RuntimeError("Missing serialized ML assets! Please verify Part 3 completed.")
        
    with open(m_path, "rb") as f: 
        model = pickle.load(f)
    with open(v_path, "rb") as f: 
        vectorizer = pickle.load(f)
    print("Production classification array mapped into memory successfully.")

@app.get("/")
def structural_health_handshake():
    """Basic landing path confirmation standard for testing server availability."""
    return {"status": "Online", "mode": "Production Containerized Environment"}

@app.post("/predict")
def score_incoming_payload(payload: TextSubmission):
    """Processes incoming runtime user strings and returns formal classification metrics."""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input processing string cannot be null.")
    
    transformed_matrix = vectorizer.transform([payload.text])
    
    probabilities = model.predict_proba(transformed_matrix)[0]
    prediction = int(model.predict(transformed_matrix)[0])
    
    return {
        "text_parsed": payload.text,
        "classification": "Toxic" if prediction == 1 else "Safe / Non-Toxic",
        "confidence_distributions": {
            "safe_probability": round(float(probabilities[0]), 4),
            "toxic_probability": round(float(probabilities[1]), 4)
        }
    }
