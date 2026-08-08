"""
Training script for the Speech Emotion Recognition model.

Usage:
    python -m src.train
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from src.config import (
    PRETRAINED_MODEL_NAME,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE_HEAD,
    LEARNING_RATE_FINETUNE,
    WEIGHT_DECAY,
    MODEL_SAVE_DIR,
    MODEL_SAVE_PATH,
)
from src.dataset import load_crema_d, SpeechEmotionDataset, collate_fn
from src.model import SpeechEmotionModel


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            if train:
                optimizer.zero_grad()

            logits = model(input_values)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def train():
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained(PRETRAINED_MODEL_NAME)
    raw_dataset = load_crema_d()

    train_dataset = SpeechEmotionDataset(raw_dataset["train"], processor)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    model = SpeechEmotionModel().to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: train the classifier head with the encoder frozen.
    model.freeze_encoder()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_HEAD,
        weight_decay=WEIGHT_DECAY,
    )

    print("Phase 1: training classifier head (encoder frozen)")
    for epoch in range(NUM_EPOCHS):
        loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS}: loss = {loss:.4f}")

    # Phase 2: unfreeze and fine-tune end-to-end with a smaller LR.
    model.unfreeze_encoder()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=WEIGHT_DECAY
    )

    print("Phase 2: fine-tuning full model (encoder unfrozen)")
    for epoch in range(NUM_EPOCHS):
        loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS}: loss = {loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
