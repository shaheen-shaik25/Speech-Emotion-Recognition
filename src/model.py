"""
Wav2Vec2-based speech emotion classifier.
"""

import torch.nn as nn
from transformers import Wav2Vec2Model

from src.config import (
    PRETRAINED_MODEL_NAME,
    NUM_CLASSES,
    CLASSIFIER_HIDDEN_SIZE,
    HIDDEN_DROPOUT,
)


class SpeechEmotionModel(nn.Module):
    """
    Wav2Vec2 encoder + mean-pooling + a small MLP classification head.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained_model=PRETRAINED_MODEL_NAME):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)

        hidden_size = self.wav2vec2.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, CLASSIFIER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(HIDDEN_DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN_SIZE, num_classes),
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def freeze_encoder(self):
        """Freeze the Wav2Vec2 encoder so only the classifier head trains."""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the Wav2Vec2 encoder for full fine-tuning."""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
