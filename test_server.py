from flask import Flask, request, render_template
from flask_cors import CORS
from datetime import datetime
import time

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

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=42069, debug=True)
