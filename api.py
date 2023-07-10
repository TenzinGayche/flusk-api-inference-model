from flask import Flask, request, jsonify
from flask_cors import CORS
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

model_id = 'openpecha/wav2vec2-large-xlsr-tibetan'
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file']
    audio_input, _ = sf.read(file)
    inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = processor.decode(predicted_ids[0])

    return jsonify({'transcript': transcript})

if __name__ == '__main__':
    app.run(port="5000")
