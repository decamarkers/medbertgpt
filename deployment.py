import tensorflow as tf
import numpy as np
import pandas as pd
# import streamlit as st
import re
import os
import csv
from tqdm import tqdm
import faiss
from nltk.translate.bleu_score import sentence_bleu
from datetime import datetime

def decontractions(phrase):
  """
    decontracted takes text and convert contractions into natural form.
    ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490
  """
  # specific
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)
  phrase = re.sub(r"won\’t", "will not", phrase)
  phrase = re.sub(r"can\’t", "can not", phrase)

  # general
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)

  phrase = re.sub(r"n\’t", " not", phrase)
  phrase = re.sub(r"\’re", " are", phrase)
  phrase = re.sub(r"\’s", " is", phrase)
  phrase = re.sub(r"\’d", " would", phrase)
  phrase = re.sub(r"\’ll", " will", phrase)
  phrase = re.sub(r"\’t", " not", phrase)
  phrase = re.sub(r"\’ve", " have", phrase)
  phrase = re.sub(r"\’m", " am", phrase)

  return phrase


def preprocess(text):
  # convert all the text into lower letters
  # remove the words betweent brakets ()
  # remove these characters: {'$', ')', '?', '"', '’', '.',  '°', '!', ';', '/', "'", '€', '%', ':', ',', '('}
  # replace these spl characters with space: '\u200b', '\xa0', '-', '/'
  
  text = text.lower()
  text = decontractions(text)
  text = re.sub('[$)\?"’.°!;\'€%:,(/]', '', text)
  text = re.sub('\u200b', ' ', text)
  text = re.sub('\xa0', ' ', text)
  text = re.sub('-', ' ', text)
  return text


#importing bert tokenizer and loading the trained question embedding extractor model
#importing gpt2 tokenizer and loading the trained gpt2 model
from transformers import AutoTokenizer, TFGPT2Model, GPT2Tokenizer, TFGPT2LMHeadModel
biobert_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/BioRedditBERT-uncased")
question_extractor_model1=tf.keras.models.load_model('question_extractor_model_2_11') #! loading model

gpt2_tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
tf_gpt2_model=TFGPT2LMHeadModel.from_pretrained("tf_gpt2_model_2_1019") #! loading gpt2 pretrained

#preparing the faiss search
qa=pd.read_pickle('./qa_gpt_inbuild_cleaned.pkl')
question_bert = qa["Q_FFNN_embeds"].tolist()
answer_bert = qa["A_FFNN_embeds"].tolist()
question_bert = np.array(question_bert)
answer_bert = np.array(answer_bert)
question_bert = question_bert.astype('float32')
answer_bert = answer_bert.astype('float32')
answer_index = faiss.IndexFlatIP(answer_bert.shape[-1])
question_index = faiss.IndexFlatIP(question_bert.shape[-1])
answer_index.add(answer_bert)
question_index.add(question_bert)

print('finished initializing')

#defining function to prepare the data for gpt inference
#https://github.com/ash3n/DocProduct

def preparing_gpt_inference_data(gpt2_tokenizer,question,question_embedding):
  topk=20
  scores,indices=answer_index.search(question_embedding.astype('float32'), topk)
  q_sub=qa.iloc[indices.reshape(20)]
  
  line = '`QUESTION: %s `ANSWER: ' % (question)
  encoded_len=len(gpt2_tokenizer.encode(line))
  for i in q_sub.iterrows():
    line='`QUESTION: %s `ANSWER: %s ' % (i[1]['question'],i[1]['answer']) + line
    line=line.replace('\n','')
    encoded_len=len(gpt2_tokenizer.encode(line))
    if encoded_len>=1024:
      break
  return gpt2_tokenizer.encode(line)[-1024:]



#function to generate answer given a question and the required answer length

def give_answer(question,answer_len):
  preprocessed_question=preprocess(question)
  question_len=len(preprocessed_question.split(' '))
  truncated_question=preprocessed_question
  if question_len>500:
    truncated_question=' '.join(preprocessed_question.split(' ')[:500])
  encoded_question= biobert_tokenizer.encode(truncated_question)
  max_length=512
  padded_question=tf.keras.preprocessing.sequence.pad_sequences([encoded_question], maxlen=max_length, padding='post')
  question_mask=[[1 if token!=0 else 0 for token in question] for question in padded_question]
  embeddings=question_extractor_model1({'question':np.array(padded_question),'question_mask':np.array(question_mask)})
  gpt_input=preparing_gpt_inference_data(gpt2_tokenizer,truncated_question,embeddings.numpy())
  mask_start = len(gpt_input) - list(gpt_input[::-1]).index(4600) + 1
  input=gpt_input[:mask_start+1]
  if len(input)>(1024-answer_len):
    input=input[-(1024-answer_len):]
  gpt2_output=gpt2_tokenizer.decode(tf_gpt2_model.generate(input_ids=tf.constant([np.array(input)]),max_length=1024,temperature=0.7)[0])
  answer=gpt2_output.rindex('`ANSWER: ')
  return gpt2_output[answer+len('`ANSWER: '):]  




def remove_repeating_phrases(text):
  words = text.split()
  n = len(words)

  # For every possible phrase length
  for phrase_len in range(n, 0, -1):
    i = 0

    # Move through words list by stepping phrase_len each time
    while i + phrase_len * 2 <= n:
      # If we find a repeated phrase, remove it
      if words[i:i+phrase_len] == words[i+phrase_len:i+2*phrase_len]:
        for _ in range(phrase_len):
            words.pop(i + phrase_len)
        n = len(words)
      else:
        i += 1

  return ' '.join(words)


# ----------------------------------------------------------------
# Flask server for usage
# ----------------------------------------------------------------

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def hello():
  return render_template("index.html")

@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
  req = request.get_json()
  question = req["question"]
  expected_answer = req["expected_answer"]
  start_time = datetime.now()
  generated_answer=remove_repeating_phrases(give_answer(question, 50))
  reference = [expected_answer.split(' ')]
  candidate = generated_answer.split(' ')
  score = {
    "1_gram": sentence_bleu(reference, candidate, weights=(1,0,0,0)),
    "2_gram": sentence_bleu(reference, candidate, weights=(0,1,0,0)),
    "3_gram": sentence_bleu(reference, candidate, weights=(0,0,1,0)),
    "4_gram": sentence_bleu(reference, candidate, weights=(0,0,0,1)),
  }
  cumulative_score = {
    "2_gram": sentence_bleu(reference, candidate, weights=(0.5,0.5,0,0)),
    "3_gram": sentence_bleu(reference, candidate, weights=(0.33,0.33,0.33,0)),
    "4_gram": sentence_bleu(reference, candidate, weights=(0.25,0.25,0.25,0.25)),
  }
  end_time = datetime.now()
  time_taken = (end_time - start_time).total_seconds()
  return jsonify({
    "question": question,
    "expected_answer": expected_answer,
    "generated_answer": generated_answer,
    "evaluation_score": score,
    "evaluation_score_cumulative": cumulative_score,
    "time_taken": time_taken,
  }), 200

@app.route("/process-question", methods=["POST"])
def process_question():
  question = request.form.get("question")
  start_time = datetime.now()
  answer = remove_repeating_phrases(give_answer(question, 50))
  end_time = datetime.now()
  time_taken = (end_time - start_time).total_seconds()
  return f"""
    <div class="response-block">
      <h2>{question}</h2>
      <p>{answer}</p>
      <i>Time Taken: {time_taken}s</i>
    </div>
  """

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=42069, debug=True)
