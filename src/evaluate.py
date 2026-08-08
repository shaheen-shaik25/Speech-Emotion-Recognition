"""
Evaluation script: computes accuracy, precision/recall/F1, a classification
report, and a confusion matrix for the trained model.

Usage:
    python -m src.evaluate
"""

import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.config import (
    PRETRAINED_MODEL_NAME,
    BATCH_SIZE,
    LABELS,
    MODEL_SAVE_PATH,
    CONFUSION_MATRIX_PATH,
    CLASSIFICATION_REPORT_PATH,
)
from src.dataset import load_crema_d, SpeechEmotionDataset, collate_fn
from src.model import SpeechEmotionModel


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2Processor.from_pretrained(PRETRAINED_MODEL_NAME)
    raw_dataset = load_crema_d()

    test_dataset = SpeechEmotionDataset(raw_dataset["test"], processor)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = SpeechEmotionModel().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"]

            logits = model(input_values)
            preds = torch.argmax(logits, dim=-1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    report = classification_report(y_true, y_pred, target_names=LABELS)
    print(report)

    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write(report)

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=[l.capitalize() for l in LABELS]
    )
    plt.title("Speech Emotion Recognition Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.show()


if __name__ == "__main__":
    evaluate()
