"""
DocFusion AI - Text-to-Speech client

Local-side helper used by app.py.
The actual TTS inference runs in the separate Colab FastAPI backend.
"""

import os
import tempfile
import time

import requests


TTS_COLAB_API = os.getenv(
    "DOCFUSION_TTS_COLAB_API",
    "https://recolor-outshine-chest.ngrok-free.dev"
).rstrip("/")


TTS_MODEL_NAMES = {
    "mms_tts_eng": "MMS-TTS English",
    "speecht5": "SpeechT5",
}


def generate_speech(text, model="mms_tts_eng"):
    """
    Send text to the TTS Colab backend and save the returned WAV file locally.

    Returns:
        tuple[str, float]: local audio filepath and backend processing time.
    """

    if not text or not text.strip():
        raise ValueError("No text was provided for speech generation.")

    if not TTS_COLAB_API or "PASTE_TTS_NGROK_URL_HERE" in TTS_COLAB_API:
        raise RuntimeError(
            "TTS Colab API URL is not configured. "
            "Set DOCFUSION_TTS_COLAB_API to your TTS ngrok URL."
        )

    model = model if model in TTS_MODEL_NAMES else "mms_tts_eng"

    start = time.perf_counter()

    response = requests.post(
        f"{TTS_COLAB_API}/speech",
        json={
            "text": text,
            "model": model,
        },
        timeout=600,
    )

    if not response.ok:
        try:
            error_data = response.json()
            message = error_data.get("error", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(f"TTS backend error: {message}")

    content_type = response.headers.get("content-type", "")
    if "audio/" not in content_type:
        try:
            data = response.json()
            raise RuntimeError(
                data.get("error", "The TTS backend did not return audio.")
            )
        except ValueError:
            raise RuntimeError(
                "The TTS backend returned an unexpected response."
            )

    backend_time_header = response.headers.get("X-Processing-Time")
    try:
        backend_time = float(backend_time_header)
    except (TypeError, ValueError):
        backend_time = time.perf_counter() - start

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix="docfusion_speech_",
        delete=False,
    ) as audio_file:
        audio_file.write(response.content)
        audio_path = audio_file.name

    return audio_path, backend_time