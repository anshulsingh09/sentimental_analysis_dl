from contextlib import asynccontextmanager
import pickle
import re

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from keras.utils import pad_sequences
from pydantic import BaseModel, Field

model_path = "Artifacts/BIGRU_Model.keras"
tokenizer_path = "Artifacts/tokenizer.pkl"
max_sequence_length = 50
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Holds the loaded model and tokenizer while the server is running
dl_model = {}


# ---------- PREPROCESSING ----------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- SCHEMAS ----------
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"},
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------- APP LIFECYCLE ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")
    dl_model["BiGRU"] = load_model(model_path)
    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print("Model loaded successfully...")

    yield  # server is running and handling requests

    dl_model.clear()


app = FastAPI(lifespan=lifespan)

# CORS (Cross Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------- ROUTES ----------
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    model = dl_model.get("BiGRU")
    tokenizer = dl_model.get("Tokenizer")

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later.",
        )

    cleaned_text = preprocess_text(text_input.text)

    # texts_to_sequences expects a LIST of texts
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(
        sequences,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(padded, verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    all_probabilities = {
        label: float(prob) for label, prob in zip(emotion_labels, probabilities)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_index],
        confidence=float(probabilities[top_index]),
        all_probabilities=all_probabilities,
    )
