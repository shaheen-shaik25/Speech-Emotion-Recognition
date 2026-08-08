"""
Dataset loading and PyTorch Dataset/DataLoader utilities for CREMA-D.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from src.config import HF_DATASET_NAME, SAMPLE_RATE


def load_crema_d():
    """Load the CREMA-D dataset from the Hugging Face Hub."""
    dataset = load_dataset(HF_DATASET_NAME)
    return dataset


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
