from flask import Flask, request, render_template
from flask_cors import CORS
from datetime import datetime
import time
import whisper

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def hello():
  return render_template("index.html")

@app.route("/process_question", methods=["POST"])
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

@app.route("/convertSpeechToText/<string:recordingFilepath>", methods=["GET"])
def convertSpeechToText(recordingFilepath):
  # pip install -U openai-whisper
  # May need FFMPEG? https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
  print(recordingFilepath)

  # Set model filepath
  # Default code to download: model = whisper.load_model("base")
  # After downloading, model will be stored in C:\Users\<username>\.cache\whisper\<model>
  model = whisper.load_model("base.pt")

  # load audio and pad/trim it to fit 30 seconds
  # audio should be mp3/wav? Need to check, m4a seems to work
  audio = whisper.load_audio("flu symptoms.m4a") # Replace with recordingFilepath
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
