# predict.py
import sys
import json
import tempfile
import requests
import whisperx

def predict(audio_url: str, model_size: str = "base"):
    # Download the audio file from the given URL
    response = requests.get(audio_url)
    if response.status_code != 200:
        raise Exception("Could not download audio file.")
    
    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(response.content)
        temp_audio.flush()
        audio_path = temp_audio.name

    # Transcribe using WhisperX
    # (Here we use the python API; adjust if needed based on your use-case)
    try:
        result = whisperx.transcribe(audio_path, model=model_size)
    except Exception as e:
        raise Exception(f"Transcription failed: {e}")
    
    return result

if __name__ == '__main__':
    # Replicate sends input as JSON via STDIN.
    # For example: {"audio_url": "https://example.com/audio.wav", "model_size": "base"}
    input_data = json.loads(sys.stdin.read())
    audio_url = input_data.get("audio_url")
    model_size = input_data.get("model_size", "base")
    
    if not audio_url:
        raise ValueError("Input must include an 'audio_url' key.")

    output = predict(audio_url, model_size=model_size)
    # Print the result as JSON so Replicate can capture it.
    print(json.dumps(output))
