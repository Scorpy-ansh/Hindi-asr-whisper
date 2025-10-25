import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"
from flask import Response
import json
from flask import Flask, request, jsonify
import torch, torchaudio, tempfile
from transformers import pipeline

app = Flask(__name__)

DEVICE = 0 if torch.cuda.is_available() else -1
asr = pipeline(
    "automatic-speech-recognition",
    model="collabora/whisper-small-hindi",
    device=DEVICE,
    framework="pt"  
)

def load_audio(file, target_sr=16000):
    """Load uploaded audio and resample to 16 kHz mono."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        wav, sr = torchaudio.load(tmp.name)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return {"array": wav.squeeze().numpy(), "sampling_rate": target_sr}

@app.route('/')
def home():
    return "✅ Hindi ASR Flask API is running. Use POST /predict with an audio file."

@app.route('/predict', methods=['POST'])
def predict():
    """Receive an audio file and return its Hindi transcription."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    audio = load_audio(file)
    out = asr(audio, generate_kwargs={"language": "hi", "task": "transcribe"})
    return Response(json.dumps({"transcript": out["text"]}, ensure_ascii=False), mimetype='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(debug=True)
