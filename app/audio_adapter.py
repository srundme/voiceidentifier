#!/usr/bin/env python
# ---------------------------------------------------------------
# app/audio_adapter.py
# ---------------------------------------------------------------
# Audio Adapter Layer for the Voice Authentication API.
#
# Provides a unified interface for converting different audio
# input formats (WAV, raw PCM 8kHz) into a standardized WAV
# file suitable for the downstream VAD and embedding pipeline.
#
# Supported formats:
#   - WAV  (standard file upload, used by Swagger and existing clients)
#   - PCM  (raw 16-bit signed little-endian, 8kHz mono, from telephony)
# ---------------------------------------------------------------

import wave
import numpy as np
from pathlib import Path


# Default PCM telephony parameters
PCM_DEFAULT_SAMPLE_RATE = 8000
PCM_DEFAULT_CHANNELS = 1
PCM_DEFAULT_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


def pcm_to_wav(
    pcm_bytes: bytes,
    output_path: Path,
    sample_rate: int = PCM_DEFAULT_SAMPLE_RATE,
    channels: int = PCM_DEFAULT_CHANNELS,
    sample_width: int = PCM_DEFAULT_SAMPLE_WIDTH
) -> Path:
    """
    Convert raw PCM bytes to a standard WAV file on disk.

    Args:
        pcm_bytes (bytes): Raw PCM audio payload (16-bit signed little-endian).
        output_path (Path): Destination path for the generated WAV file.
        sample_rate (int): Sample rate of the PCM data. Default 8000 Hz.
        channels (int): Number of audio channels. Default 1 (mono).
        sample_width (int): Bytes per sample. Default 2 (16-bit).

    Returns:
        Path: The path to the written WAV file.

    Raises:
        ValueError: If pcm_bytes is empty.
        RuntimeError: If writing the WAV file fails.
    """
    if not pcm_bytes:
        raise ValueError("PCM payload is empty. Cannot convert to WAV.")

    try:
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return output_path
    except Exception as exc:
        raise RuntimeError(f"Failed to write WAV from PCM data: {exc}") from exc


def pcm_bytes_to_numpy(
    pcm_bytes: bytes,
    sample_rate: int = PCM_DEFAULT_SAMPLE_RATE
):
    """
    Decode raw PCM bytes directly into a NumPy float32 waveform.

    Useful when you want to avoid writing to disk.

    Args:
        pcm_bytes (bytes): Raw PCM audio payload (16-bit signed little-endian).
        sample_rate (int): Sample rate of the PCM data.

    Returns:
        Tuple[np.ndarray, int]: (waveform float32 array normalised to [-1, 1], sample_rate)

    Raises:
        ValueError: If pcm_bytes is empty or has odd length.
    """
    if not pcm_bytes:
        raise ValueError("PCM payload is empty.")

    if len(pcm_bytes) % 2 != 0:
        raise ValueError(
            f"PCM payload length ({len(pcm_bytes)}) is not a multiple of 2. "
            "Expected 16-bit (2-byte) samples."
        )

    # Interpret bytes as 16-bit signed integers, then normalise to float32 [-1, 1]
    int_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    float_samples = int_samples.astype(np.float32) / 32768.0
    return float_samples, sample_rate


def validate_wav_file(file_path: Path) -> dict:
    """
    Read basic metadata from a WAV file without loading the full audio.

    Args:
        file_path (Path): Path to the WAV file.

    Returns:
        dict: Metadata dict with keys: channels, sample_rate, sample_width,
              n_frames, duration_seconds.

    Raises:
        RuntimeError: If the file cannot be opened as WAV.
    """
    try:
        with wave.open(str(file_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            duration_seconds = n_frames / float(sample_rate) if sample_rate > 0 else 0.0
        return {
            "channels": channels,
            "sample_rate": sample_rate,
            "sample_width": sample_width,
            "n_frames": n_frames,
            "duration_seconds": round(duration_seconds, 3)
        }
    except wave.Error as exc:
        raise RuntimeError(f"Invalid or unsupported WAV file '{file_path}': {exc}") from exc
