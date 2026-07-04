#!/usr/bin/env python
# ---------------------------------------------------------------
# app/gender_service.py
# ---------------------------------------------------------------
# Provides an independent module to predict speaker gender using
# a pretrained Hugging Face transformers model.
# ---------------------------------------------------------------

import traceback
from typing import Dict, Union

# Global model cache to ensure we only load it once
_gender_pipeline = None

def _get_pipeline():
    """Lazily load and cache the gender classification pipeline."""
    global _gender_pipeline
    if _gender_pipeline is None:
        try:
            from transformers import pipeline
            # Using a well-supported, pretrained gender classification model
            _gender_pipeline = pipeline(
                task="audio-classification", 
                model="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
            )
        except Exception as e:
            print(f"[ERROR] Failed to load gender classification pipeline: {e}")
            raise
    return _gender_pipeline


def predict_gender(audio_path: str) -> Dict[str, Union[str, float]]:
    """
    Predicts the gender of the speaker in the given audio file.
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file.
        
    Returns
    -------
    dict
        {"gender": "Male" | "Female" | "Unknown", "confidence": float}
    """
    try:
        classifier = _get_pipeline()
        
        # The pipeline parses the audio from the file path directly
        # Example output: [{'score': 0.99, 'label': 'male'}, {'score': 0.01, 'label': 'female'}]
        results = classifier(audio_path)
        
        if not results:
            return {"gender": "Unknown", "confidence": 0.0}
            
        best_prediction = results[0]
        label = str(best_prediction.get("label", "Unknown")).strip().lower()
        score = float(best_prediction.get("score", 0.0))
        
        if label == "male":
            gender = "Male"
        elif label == "female":
            gender = "Female"
        else:
            gender = "Unknown"
            
        return {
            "gender": gender,
            "confidence": round(score * 100, 2)
        }
        
    except Exception as e:
        print("[ERROR] predict_gender encountered an exception:")
        traceback.print_exc()
        # Fail gracefully without breaking authentication
        return {"gender": "Unknown", "confidence": 0.0}

# ---------------------------------------------------------------
# Warm up the model on import (like embedding.py) to prevent 
# timeouts during the first request
# ---------------------------------------------------------------
try:
    _get_pipeline()
except Exception:
    pass
