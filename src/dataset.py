"""
Dataset loading and PyTorch Dataset/DataLoader utilities for CREMA-D.

Note: AbstractTTS/CREMA-D on the Hugging Face Hub ships as a single
"train" split with a string label column called `major_emotion`
(values: anger, disgust, fear, happy, neutral, sad) rather than an
integer `label` column, and with no pre-made test split. This module
maps the string label to the integer ids in src.config.LABEL2ID and
carves out a held-out test split ourselves.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from src.config import HF_DATASET_NAME, SAMPLE_RATE, LABEL2ID, TRAIN_TEST_SPLIT, RANDOM_SEED


def load_crema_d():
    """
    Load the CREMA-D dataset from the Hugging Face Hub and return a
    dict-like object with "train" and "test" splits, each carrying an
    integer `label` column derived from `major_emotion`.
    """
    dataset = load_dataset(HF_DATASET_NAME)

    # This mirror only ships a single "train" split.
    full = dataset["train"]

    def add_label(example):
        example["label"] = LABEL2ID[example["major_emotion"].lower()]
        return example

    full = full.map(add_label)

    split = full.train_test_split(test_size=TRAIN_TEST_SPLIT, seed=RANDOM_SEED)
    return split  # dict-like with "train" and "test" keys


class SpeechEmotionDataset(Dataset):
    """
    Wraps a Hugging Face audio dataset split and a Wav2Vec2Processor
    to produce model-ready tensors.
    """

    def __init__(self, hf_split, processor, sample_rate=SAMPLE_RATE):
        self.dataset = hf_split
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio = example["audio"]

        inputs = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            padding=True,
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(example["label"], dtype=torch.long),
        }


def collate_fn(batch):
    """
    Pad variable-length input_values within a batch to the same length.
    """
    input_values = [item["input_values"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    max_len = max(v.shape[0] for v in input_values)
    padded = torch.zeros(len(input_values), max_len)
    for i, v in enumerate(input_values):
        padded[i, : v.shape[0]] = v

    return {"input_values": padded, "labels": labels}
