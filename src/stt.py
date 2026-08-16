"""
Phase 6: Speech-to-text using ElevenLabs (Scribe model).
"""

import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


def transcribe_audio(file_path: str) -> str:
    """
    Transcribes a speech audio file to text using ElevenLabs Scribe.
    Auto-detects language, works well with Hindi.
    """
    with open(file_path, "rb") as f:
        transcription = eleven_client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
        )
    return transcription.text