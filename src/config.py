"""
Central configuration for the Speech Emotion Recognition project.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_SAVE_DIR = os.path.join(MODELS_DIR, "wav2vec2_emotion")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "model.pt")

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
TRAINING_CURVE_PATH = os.path.join(RESULTS_DIR, "training_curve.png")
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
HF_DATASET_NAME = "AbstractTTS/CREMA-D"
SAMPLE_RATE = 16000

LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
PRETRAINED_MODEL_NAME = "facebook/wav2vec2-base"
HIDDEN_DROPOUT = 0.3
CLASSIFIER_HIDDEN_SIZE = 256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE_HEAD = 1e-4       # used while encoder is frozen
LEARNING_RATE_FINETUNE = 1e-5   # used once encoder is unfrozen
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

DEVICE = "cuda"  # overridden at runtime if unavailable, see src/train.py
