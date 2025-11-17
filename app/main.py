import os
import tempfile
import time
import traceback
from typing import List, Optional

import torch
import librosa
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from transformers import pipeline

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
DEVICE = 0 if torch.cuda.is_available() else -1

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "NbAiLab/nb-whisper-large-distil-turbo-beta")
DEFAULT_LANG = "no"
LANG_CHOICES = ["no", "nn", "en"]

# Chunking + decoding settings
CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", 28))   # 28s recommended
NUM_BEAMS = int(os.getenv("NUM_BEAMS", 5))          # Higher accuracy, slower

app = FastAPI(title="NB-Whisper Simple ASR", version="0.1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ----------------------------------------------------------
# Load ASR model ONCE at startup
# ----------------------------------------------------------

print("Loading HuggingFace ASR pipeline...")
asr = pipeline(
    "automatic-speech-recognition",
    model=DEFAULT_MODEL,
    device=DEVICE,
)


# ----------------------------------------------------------
# Pydantic models (for API responses)
# ----------------------------------------------------------

class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    filename: str
    language: str
    text: str
    segments: Optional[List[Segment]] = None


# ----------------------------------------------------------
# Core transcription (runs in thread pool)
# ----------------------------------------------------------

def run_asr(path: str, lang: str):

    lang = lang or DEFAULT_LANG

    output = asr(
        path,
        chunk_length_s=CHUNK_LENGTH,
        return_timestamps=True,
        generate_kwargs={
            "task": "transcribe",
            "language": lang,
            "num_beams": NUM_BEAMS,
        },
    )

    text = output["text"]
    chunks = output.get("chunks", [])

    segments: List[Segment] = []
    for c in chunks:
        ts = c.get("timestamp")
        if ts:
            segments.append(
                Segment(start=float(ts[0]), end=float(ts[1]), text=c["text"].strip())
            )

    return text, segments


# ----------------------------------------------------------
# API endpoint
# ----------------------------------------------------------

@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_api(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        text, segments = await run_in_threadpool(run_simple_asr, tmp_path, lang)

        return TranscriptionResponse(
            filename=file.filename,
            language=lang,
            text=text,
            segments=segments,
        )

    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ----------------------------------------------------------
# Web UI
# ----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "languages": LANG_CHOICES,
            "selected_lang": DEFAULT_LANG,
            "result_text": None,
            "filename": None,
            "is_error": False,
            "transcribe_time": None,
            "audio_length": None,
            "word_count": None,
        },
    )


@app.post("/ui/transcribe", response_class=HTMLResponse)
async def transcribe_ui(
    request: Request,
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    result_text = None
    is_error = False
    transcribe_time = None
    audio_length = None
    word_count = None

    try:
        data = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(data)

        # Get audio duration using librosa
        try:
            y, sr = librosa.load(tmp_path, sr=None)
            audio_length = round(len(y) / sr, 1)
        except Exception:
            audio_length = None

        start_time = time.time()
        text, _segments = await run_in_threadpool(run_asr, tmp_path, lang)
        end_time = time.time()
        
        transcribe_time = round(end_time - start_time, 1)
        result_text = text.strip()
        
        # Calculate word count
        if result_text:
            word_count = len(result_text.split())

    except Exception:
        result_text = traceback.format_exc()
        is_error = True

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "languages": LANG_CHOICES,
            "selected_lang": lang,
            "result_text": result_text,
            "filename": file.filename,
            "is_error": is_error,
            "transcribe_time": transcribe_time,
            "audio_length": audio_length,
            "word_count": word_count,
        },
    )

@app.get("/debug")
async def debug():
    return {"status": "ok"}
