"""
Single-file inference utility: given a path to a WAV/MP3 file, predict
the speaker's emotion.

Usage:
    python -m src.predict path/to/audio.wav
"""

import sys
import torch
from transformers import Wav2Vec2Processor

from src.config import PRETRAINED_MODEL_NAME, LABELS, MODEL_SAVE_PATH, SAMPLE_RATE
from src.model import SpeechEmotionModel
from src.preprocessing import preprocess_audio


def load_model(device="cpu"):
    processor = Wav2Vec2Processor.from_pretrained(PRETRAINED_MODEL_NAME)

    model = SpeechEmotionModel()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    return processor, model


def predict(audio_path, device="cpu"):
    processor, model = load_model(device)

    audio, sr = preprocess_audio(audio_path, sr=SAMPLE_RATE)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values)
        probabilities = torch.softmax(logits, dim=-1)[0]
        prediction = torch.argmax(probabilities).item()

    result = {
        "emotion": LABELS[prediction],
        "confidence": round(probabilities[prediction].item() * 100, 2),
        "probabilities": {
            label: round(probabilities[i].item() * 100, 2)
            for i, label in enumerate(LABELS)
        },
    }

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_audio_file>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(f"Predicted Emotion: {result['emotion'].upper()}")
    print(f"Confidence: {result['confidence']}%")
