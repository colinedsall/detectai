from fastapi import APIRouter, HTTPException
from app.schemas.text import TextDetectRequest, TextDetectResponse
from app.services.ml_text_detector import MLTextDetector
import time

router = APIRouter()

detector = MLTextDetector(model_path="ai_detector_model.pkl")

@router.post("/text", response_model=TextDetectResponse)
async def detect_text(request: TextDetectRequest):
    start_time = time.time()
    
    model_output = detector.predict(request.text)
    
    end_time = time.time()
    processing_time_ms = int((end_time - start_time) * 1000)

    # --- START OF CORRECTIONS ---

    # Calculate unique_words from the model output
    word_count = model_output["features"]["word_count"]
    unique_ratio = model_output["features"]["unique_word_ratio"]
    unique_words = int(word_count * unique_ratio) # FIX 3

    final_response = {
        "prediction": model_output["prediction"],
        "probability_ai": model_output["probability_ai"], # FIX 1: Renamed from "probability"
        "confidence": model_output["confidence"],
        "methods": [
            {
                "name": model_output["method"],
                "prediction": model_output["prediction"],
                "confidence": model_output["confidence"],
                "explanation": ", ".join(model_output["explanations"]), # FIX 2: Added explanations
                "weight": 1.0
            }
        ],
        "metadata": {
            "word_count": word_count,
            "unique_words": unique_words, # FIX 3: Added unique_words
            "character_count": len(request.text),
            "unique_word_ratio": unique_ratio,
            "avg_word_length": model_output["features"]["avg_word_length"],
            "source_type": "text",
            "processing_time_ms": processing_time_ms
        }
    }
    
    # --- END OF CORRECTIONS ---
    
    return final_response