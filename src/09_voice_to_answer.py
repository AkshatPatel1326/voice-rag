"""
Phase 6: Full voice-to-answer pipeline.
Voice file -> transcription -> retrieval -> guarded generation -> answer

Run this from the project root:
    python src/09_voice_to_answer.py path/to/audio_file
"""

import sys
from stt import transcribe_audio
from harness import RAGHarness, QueryRequest

if len(sys.argv) < 2:
    print("Usage: python src/09_voice_to_answer.py <path_to_audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]

print(f"Step 1: Transcribing audio -> {audio_path}")
query_text = transcribe_audio(audio_path)
print(f"Transcribed text: {query_text}")

print("\nStep 2: Loading RAG harness...")
harness = RAGHarness()

print("\nStep 3: Running guarded pipeline...")
request = QueryRequest(query=query_text)
response = harness.run(request)

print("\n=== Final Result ===")
print(response.model_dump_json(indent=2))