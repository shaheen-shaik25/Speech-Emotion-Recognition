"""
Streamlit front-end for the Speech Emotion Recognition project.

Run with:
    streamlit run app/app.py
"""

import os
import sys

import streamlit as st
import torch
import librosa
import pandas as pd

# Allow importing from src/ when running via `streamlit run app/app.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Wav2Vec2Processor
from src.config import PRETRAINED_MODEL_NAME, LABELS, MODEL_SAVE_PATH, SAMPLE_RATE
from src.model import SpeechEmotionModel


@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(PRETRAINED_MODEL_NAME)

    model = SpeechEmotionModel(num_classes=len(LABELS))
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()

    return processor, model


st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a speech recording to predict the speaker's emotion.")

uploaded_file = st.file_uploader("Upload WAV/MP3 audio", type=["wav", "mp3"])

if uploaded_file:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
    duration = librosa.get_duration(y=audio, sr=sr)

    st.audio(uploaded_file)

    st.subheader("Audio information")
    st.write(f"Sampling Rate: {sr} Hz")
    st.write(f"Duration: {duration:.2f} seconds")
    st.write("Channels: Mono")

    processor, model = load_model()

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values)
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()

    emotion = LABELS[prediction]
    confidence = probabilities[0][prediction].item() * 100

    st.success(f"Emotion: {emotion.capitalize()}")
    st.metric("Confidence", f"{confidence:.2f}%")

    st.subheader("Prediction breakdown")
    probs_df = pd.DataFrame(
        {
            "Emotion": [l.capitalize() for l in LABELS],
            "Probability (%)": [
                round(probabilities[0][i].item() * 100, 2) for i in range(len(LABELS))
            ],
        }
    ).sort_values("Probability (%)", ascending=False)

    st.bar_chart(probs_df.set_index("Emotion"))

    os.remove(temp_path)
