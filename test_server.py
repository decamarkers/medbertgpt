from flask import Flask, request, render_template
from flask_cors import CORS
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def hello():
  return render_template("index.html")

@app.route("/process-question", methods=["POST"])
def process_question():
  question = request.form.get("question")
  start_time = datetime.now()
  answer = "Example test response with ~1s delay, given the test question: " + question
  time.sleep(1)
  end_time = datetime.now()
  time_taken = (end_time - start_time).total_seconds()
  return f"""
    <div class="response-block">
      <h3>{question}</h3>
      <p>{answer}</p>
      <p><i>{time_taken}</i></p>
    </div>
  """

# import speech_recognition as sr
# import wavio
# import numpy as np
# from io import BytesIO

# @app.route("/process-speech", methods=["POST"])
# def process_speech():
#   audio = request.files.get("audio")
#   with open("temp_audio.wav", "wb") as f:
#     f.write(audio.read())
#   recognizer = sr.Recognizer()
#   with sr.AudioFile("temp_audio.wav") as source:
#     audio_data = recognizer.record(source)
#     try:
#       text = recognizer.recognize_google(audio_data)
#       print("Recognized Text:", text)
#     except sr.UnknownValueError:
#       print("Google Speech Recognition could not understand the audio.")
#     except sr.RequestError as e:
#       print(f"Could not request results from Google Speech Recognition service; {e}")

# @app.route("/process-deez", methods=["POST"])
# def nuts():
#   audio = request.files.get("audio")
#   with open("temp_audio.wav", "wb") as f:
#     f.write(audio.read())
#   file = BytesIO()
#   file.write(audio.read())
#   file.seek(0)
#   r = sr.Recognizer()
#   with sr.AudioFile(file) as source:
#     audio_data = r.record(source)
#   result = r.recognize_google(audio_data=audio_data, language="en-US", show_all=True)
#   return jsonify({"text": result})

import whisper

@app.route("/convertSpeechToText/<string:recordingFilepath>", methods=["GET"])
def convertSpeechToText(recordingFilepath):
  # pip install -U openai-whisper
  # Install FFMPEG https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/

  # Set model filepath
  # Default code to download: model = whisper.load_model("base")
  # After downloading, model will be stored in C:\Users\<username>\.cache\whisper\<model>
  model = whisper.load_model("base")

  # audio should be mp3/wav? Need to check, m4a seems to work
  audio = whisper.load_audio("flu symptoms.m4a") # Replace with recordingFilepath

  # load audio and pad/trim it to fit 30 seconds
  audio = whisper.pad_or_trim(audio)

  # make log-Mel spectrogram and move to the same device as the model
  mel = whisper.log_mel_spectrogram(audio).to(model.device)
  
  # detect the spoken language
  _, probs = model.detect_language(mel)
  print(f"Detected language: {max(probs, key=probs.get)}")
  # decode the audio
  options = whisper.DecodingOptions(fp16 = False)
  result = whisper.decode(model, mel, options)
  # print the recognized text
  print(result.text)
  return(result.text)

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=42069, debug=True)
