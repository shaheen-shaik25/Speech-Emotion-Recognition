"""
Audio preprocessing and signal-processing feature extraction utilities.
"""

import numpy as np
import librosa

from src.config import SAMPLE_RATE


def preprocess_audio(path, sr=SAMPLE_RATE):
    """
    Load an audio file, convert to mono, resample to the target
    sampling rate, and peak-normalize the waveform.
    """
    audio, sr = librosa.load(path, sr=sr, mono=True)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio, sr


def get_audio_info(path, sr=SAMPLE_RATE):
    """
    Return basic metadata about an audio file: sample rate,
    duration in seconds, and number of channels.
    """
    audio, loaded_sr = librosa.load(path, sr=sr, mono=True)
    duration = librosa.get_duration(y=audio, sr=loaded_sr)
    return {
        "sampling_rate": loaded_sr,
        "duration_seconds": round(float(duration), 2),
        "channels": 1,
    }


def compute_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=128):
    """
    Compute a log-scaled Mel spectrogram for a waveform.
    """
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def compute_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=40):
    """
    Compute MFCC features for a waveform.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def add_noise(audio, noise_factor=0.005):
    """Simple additive white-noise augmentation."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def time_stretch(audio, rate=1.1):
    """Time-stretch augmentation (rate > 1 speeds up, < 1 slows down)."""
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr=SAMPLE_RATE, n_steps=2):
    """Pitch-shift augmentation."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
