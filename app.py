from flask import Flask, render_template, request, send_from_directory
import os
import subprocess
import uuid
import speech_recognition as sr
import text2emotion as te
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Hugging Face voice emotion model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

def detect_voice_emotion_deep(wav_path):
    try:
        speech, _ = librosa.load(wav_path, sr=16000)
        inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        labels = model.config.id2label
        scores = {labels[i]: round(probs[i].item(), 3) for i in range(len(probs))}
        top_emotion = max(scores, key=scores.get)
        return top_emotion, scores
    except Exception as e:
        return f"Voice model error: {e}", {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['audio']
    uid = str(uuid.uuid4())
    raw_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid}.webm')
    wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid}.wav')
    file.save(raw_path)

    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', raw_path,
            '-acodec', 'pcm_s16le', '-ar', '16000', wav_path
        ], check=True)
    except Exception as e:
        return f"Audio conversion failed: {e}"

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
    except Exception as e:
        text = f"Speech recognition error: {e}"

    emotions = te.get_emotion(text)
    dominant_emotion = max(emotions, key=emotions.get) if emotions else "None"
    voice_emotion, voice_scores = detect_voice_emotion_deep(wav_path)

    return render_template('index.html',
                           text=text,
                           emotions=emotions,
                           dominant_emotion=dominant_emotion,
                           voice_emotion=voice_emotion,
                           voice_scores=voice_scores,
                           audio_url=f'/uploads/{uid}.wav')

if __name__ == '__main__':
    app.run(debug=True)
