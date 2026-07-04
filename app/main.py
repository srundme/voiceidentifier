#!/usr/bin/env python
"""FastAPI entry point for the Voice Authentication service.

Provides a minimal health‑check endpoint. Additional routes will be added later.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import numpy as np

from app.embedding import generate_embedding, generate_embedding_from_waveform
from constants import DEFAULT_THRESHOLD

from fastapi.middleware.cors import CORSMiddleware
import vad_processor
from app import customer_manager


app = FastAPI(title="Voice Authentication API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensure required directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
SPEAKER_DB_DIR = BASE_DIR / "database" / "speakers"
SPEAKER_DB_DIR.mkdir(parents=True, exist_ok=True)

speaker_embeddings_cache: dict[str, np.ndarray] = {}

def reload_cache():
    """Helper to reload all speaker embeddings into the in-memory cache."""
    global speaker_embeddings_cache
    new_cache = {}
    for npy_path in SPEAKER_DB_DIR.glob("*.npy"):
        new_cache[npy_path.stem] = np.load(npy_path)
    speaker_embeddings_cache.clear()
    speaker_embeddings_cache.update(new_cache)

@app.on_event("startup")
async def startup_event():
    reload_cache()

@app.get("/", response_class=JSONResponse)
async def root() -> dict:
    """Root health‑check endpoint.

    Returns a simple JSON payload confirming that the API is running.
    """
    return {"message": "Voice Authentication API Running"}

@app.post("/enroll", response_class=JSONResponse)
async def enroll(
    name: str = Form(...),
    audio: UploadFile = File(...)
) -> dict:
    """Enroll a new speaker.

    Saves the uploaded audio temporarily, generates an embedding, stores it,
    and cleans up the temporary file. Returns JSON with status and details.
    """
    temp_path = TEMP_DIR / audio.filename
    try:
        # Save uploaded file to temporary location
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding
        embedding = generate_embedding(str(temp_path))
        # Save embedding to database
        save_path = SPEAKER_DB_DIR / f"{name}.npy"
        np.save(save_path, embedding)
        # Update in-memory cache
        speaker_embeddings_cache[name] = embedding
        # Cleanup temporary file
        temp_path.unlink(missing_ok=True)
        return {
            "status": "success",
            "speaker": name,
            "embedding_dimension": embedding.shape[0]
        }
    except Exception as exc:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"Error in /enroll: {exc}\\n{traceback.format_exc()}\\n")
        # Ensure temp file is removed on error
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

# Verification endpoint
@app.post("/verify", response_class=JSONResponse)
async def verify(
    name: str = Form(...),
    audio: UploadFile = File(...)
) -> dict:
    """Verify a speaker by comparing an uploaded audio embedding against the enrolled embedding."""
    # Save uploaded audio to temporary file
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from the uploaded audio
        embedding = generate_embedding(str(temp_path))
        # Load enrolled speaker embedding
        enrolled_path = SPEAKER_DB_DIR / f"{name}.npy"
        if not enrolled_path.is_file():
            raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found")
        enrolled_emb = np.load(enrolled_path)
        # Cosine similarity
        norm_a = np.linalg.norm(embedding)
        norm_b = np.linalg.norm(enrolled_emb)
        similarity = float(np.dot(embedding, enrolled_emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
        threshold = DEFAULT_THRESHOLD
        verified = similarity >= threshold
        return {
            "speaker": name,
            "similarity": round(similarity * 100, 2),
            "threshold": threshold * 100,
            "verified": verified,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        temp_path.unlink(missing_ok=True)
# Identification endpoint
@app.post("/identify", response_class=JSONResponse)
async def identify(
    audio: UploadFile = File(...)
) -> dict:
    """Identify the most similar enrolled speaker."""
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from uploaded audio
        embedding = generate_embedding(str(temp_path))
        if not speaker_embeddings_cache:
            raise HTTPException(status_code=404, detail="No enrolled speakers found")
        # Compute similarities
        best_name = "Unknown"
        best_sim = 0.0
        for name, emb in speaker_embeddings_cache.items():
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = float(np.dot(embedding, emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
            if sim > best_sim:
                best_sim = sim
                best_name = name
        threshold = DEFAULT_THRESHOLD
        identified = best_sim >= threshold
        result = {
            "predicted_speaker": best_name if identified else "Unknown",
            "similarity": round(best_sim * 100, 2),
            "identified": identified,
        }
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        temp_path.unlink(missing_ok=True)

# Authenticate endpoint (Orchestration)
@app.post("/authenticate", response_class=JSONResponse)
async def authenticate(audio: UploadFile = File(...)) -> dict:
    """Automatic Customer Voice Identity Service endpoint."""
    import time
    
    total_start = time.perf_counter()
    temp_path = TEMP_DIR / audio.filename

    try:
        # Save the uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # 1. Check if we have any enrolled speakers at all
        if not speaker_embeddings_cache:
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "warning",
                "existing_customer": False,
                "authentication_result": "NO_REGISTERED_SPEAKERS",
                "similarity": 0.0,
                "processing_time_ms": round(processing_time_ms, 2),
                "message": "No enrolled speakers found in the system."
            }

        # 2. VAD & Audio Loading
        waveform, sr = vad_processor.load_audio(temp_path)
        speech_waveform = vad_processor.remove_silence(waveform, sr)
        
        duration_seconds = len(speech_waveform) / sr
        if duration_seconds < 15.0:
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "error",
                "authentication_result": "INSUFFICIENT_AUDIO",
                "required_duration_seconds": 15.0,
                "received_duration_seconds": round(duration_seconds, 2),
                "processing_time_ms": round(processing_time_ms, 2),
                "message": "Need at least 15 seconds of clear speech."
            }

        # 3. Generate the embedding
        embedding = generate_embedding_from_waveform(speech_waveform, sr)



        # 4. Cosine similarity search
        best_name = None
        best_sim = 0.0
        
        for name, emb in speaker_embeddings_cache.items():
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = (
                float(np.dot(embedding, emb) / (norm_a * norm_b))
                if norm_a and norm_b
                else 0.0
            )
            if sim > best_sim:
                best_sim = sim
                best_name = name
                
        threshold = DEFAULT_THRESHOLD

        # 5. Customer resolution
        if best_name and best_sim >= threshold:
            # Existing Customer Match
            customer = customer_manager.update_customer(best_name)
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "success",
                "existing_customer": True,
                "customer_id": best_name,
                "customer_name": customer.get("name") if customer else None,
                "similarity": round(best_sim * 100, 2),
                "threshold": threshold,
                "authentication_result": "AUTHENTICATED",
                "call_count": customer.get("call_count", 1) if customer else 1,
                "processing_time_ms": round(processing_time_ms, 2),
                "message": "Customer authenticated successfully."
            }
        else:
            # New Customer Registration
            new_id = customer_manager.generate_customer_id()
            save_path = SPEAKER_DB_DIR / f"{new_id}.npy"
            np.save(save_path, embedding)
            speaker_embeddings_cache[new_id] = embedding
            customer = customer_manager.create_customer(new_id)
            
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "success",
                "existing_customer": False,
                "customer_id": new_id,
                "similarity": round(best_sim * 100, 2) if best_name else 0.0,
                "threshold": threshold,
                "authentication_result": "NEW_SPEAKER_ENROLLED",
                "call_count": customer.get("call_count", 1) if customer else 1,
                "processing_time_ms": round(processing_time_ms, 2),
                "message": "New customer enrolled successfully."
            }

    except Exception as exc:
        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "authentication_result": "INTERNAL_SERVER_ERROR",
            "processing_time_ms": round(processing_time_ms, 2),
            "message": str(exc)
        }
    finally:
        temp_path.unlink(missing_ok=True)
