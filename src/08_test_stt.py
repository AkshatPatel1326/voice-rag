"""
Phase 6: Test speech-to-text with a real audio file.

Run this from the project root:
    python src/08_test_stt.py path/to/your/audio.wav
"""

import sys
from stt import transcribe_audio

if len(sys.argv) < 2:
    print("Usage: python src/08_test_stt.py <path_to_audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]
print(f"Transcribing: {audio_path}")

transcript = transcribe_audio(audio_path)
print(f"\nTranscript: {transcript}")