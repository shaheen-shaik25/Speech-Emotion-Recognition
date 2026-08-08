# 🎙️ Speech Emotion Recognition using Wav2Vec2

Fine-tuning `facebook/wav2vec2-base` for 6-class speech emotion classification on the CREMA-D dataset, with a Streamlit demo app.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Librosa
- Scikit-learn
- Pandas / NumPy
- Matplotlib
- Streamlit

## Overview

```
              AUDIO
                ↓
        Audio Preprocessing
                ↓
       Resample to 16 kHz
                ↓
       Noise/Amplitude handling
                ↓
          Wav2Vec2 Encoder
                ↓
        Speech Embeddings
                ↓
       Classification Head
                ↓
       Emotion Prediction
                ↓
 ┌────────────────────────────┐
 │ Angry / Disgust / Fear     │
 │ Happy / Neutral / Sad      │
 └────────────────────────────┘
```

Example:

```
Input:  person_speech.wav
Output: Predicted Emotion: HAPPY (Confidence: 87.4%)
```

## Dataset

**CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)

- 7,442 original clips
- 91 actors
- 12 sentences
- 6 emotions: anger, disgust, fear, happy, neutral, sad
- WAV speech recordings

Loaded via the Hugging Face Hub: [`myleslinder/crema-d`](https://huggingface.co/datasets/myleslinder/crema-d)

## Model

**`facebook/wav2vec2-base`** — a self-supervised speech representation model trained on 16 kHz audio.

```
Raw Audio
    ↓
Wav2Vec2 Feature Encoder
    ↓
Transformer Encoder
    ↓
Speech Representation (768-dim)
    ↓
Mean Pooling
    ↓
Linear(768 → 256) → ReLU → Dropout
    ↓
Linear(256 → 6)
    ↓
Emotion Prediction
```

## Project Structure

```
speech-emotion-recognition/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_audio_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_wav2vec2_finetuning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app/
│   └── app.py
│
├── models/
│   └── wav2vec2_emotion/
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curve.png
│   └── classification_report.txt
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Training is GPU-intensive — Google Colab (or another CUDA-capable machine) is recommended if you don't have a local NVIDIA GPU.

### 2. Explore the data

Open `notebooks/01_data_exploration.ipynb` to inspect class balance and sample clips.

### 3. Preprocess & inspect signal features

`notebooks/02_audio_preprocessing.ipynb` walks through waveform, Mel spectrogram, and MFCC visualization — useful both as EDA and as an interview talking point:

> "I converted the raw speech waveform into a time-frequency representation using a Mel spectrogram to analyze how frequency components vary over time."

### 4. Train a baseline

`notebooks/03_baseline_model.ipynb` fits a classical MFCC + Random Forest baseline for comparison against the deep model.

### 5. Fine-tune Wav2Vec2

```bash
python -m src.train
```

Training happens in two phases:

1. **Frozen encoder** — only the classification head is trained (fast, stable).
2. **Fine-tuning** — the full Wav2Vec2 encoder is unfrozen and trained with a smaller learning rate.

> "I initially froze the pretrained Wav2Vec2 encoder and trained the classification head. After establishing a baseline, I unfroze the encoder and fine-tuned it with a lower learning rate."

### 6. Evaluate

```bash
python -m src.evaluate
```

Generates `results/classification_report.txt` and `results/confusion_matrix.png`.

### 7. Run inference on a single file

```bash
python -m src.predict path/to/audio.wav
```

### 8. Launch the demo app

```bash
streamlit run app/app.py
```

## Baseline Comparison

| Model | Accuracy | F1 |
|---|---|---|
| MFCC + Random Forest | _fill in_ | _fill in_ |
| CNN/LSTM | _fill in_ | _fill in_ |
| Wav2Vec2 (fine-tuned) | _fill in_ | _fill in_ |

*Fill in with actual experiment results — do not fabricate numbers.*

## Research Component

Base paper: **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**

```
Research Paper
      ↓
Understand Architecture
      ↓
Implement / Adapt Model
      ↓
Fine-tune on CREMA-D
      ↓
Evaluate
      ↓
Analyze Errors
      ↓
Propose Improvements
```

### Experimental Analysis

- **Experiment 1** — Frozen encoder, trainable classifier
- **Experiment 2** — Fully fine-tuned encoder + classifier
- **Experiment 3** — Learning-rate sweep (1e-4, 5e-5, 1e-5)
- **Experiment 4** — Audio augmentation (noise, time-stretch, pitch-shift)

### Error Analysis

Investigate misclassifications (e.g. Angry → Fear, Sad → Neutral, Disgust → Angry) against:

- background noise
- speaker variation
- emotion similarity
- audio duration / intensity
- pronunciation
- recording quality

## Future Work: Emotion-Aware Voice Assistant

```
User Speech
     ↓
Speech Recognition
     ↓
Emotion Detection
     ↓
Text Understanding
     ↓
Response Generation
     ↓
Emotion-Adaptive Response
```

Example: a user says *"I'm really frustrated with this!"* → emotion detected as **Angry** → the assistant adapts its response to be calmer, shorter, and more empathetic.

An extended version could combine Wav2Vec2 (emotion) with Whisper (speech-to-text) feeding into an NLP response module for a full emotion-aware speech assistant.

## License

MIT (or your preferred license).
