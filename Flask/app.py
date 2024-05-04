from urllib import request
from flask import *
import pyrebase
import os

import pandas as pd
from cdqa.utils.converters import pdf_converter
from cdqa.pipeline import QAPipeline
import joblib
import pdfplumber

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

config = {
	"apiKey": "AIzaSyBWKLy0OIHNILtOVhjk5iUy7mIdQq-MB_s",
    "authDomain": "nlpqna-b2a1e.firebaseapp.com",
    "projectId": "nlpqna-b2a1e",
	"databaseURL" : "",
    "storageBucket": "nlpqna-b2a1e.appspot.com",
    "messagingSenderId": "176135293763",
    "appId": "1:176135293763:web:8cf093edf0bd863cd0fdf6",
    "measurementId": "G-FEC330SVED"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

def load_glove_model(glove_input_file):
    word2vec_output_file = glove_input_file + '.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)
    glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    return glove_model

glove_model = load_glove_model('~/Downloads/QnA_NLP_SYR/Flask/models/glove.6B.100d.txt')

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()  
    return text

def fine_tuning_drive(question, file_name, model_type):
    local_directory = os.path.expanduser('~/Downloads/QnA_NLP_SYR/Flask/docs')
    os.makedirs(local_directory, exist_ok=True)
    local_file_path = os.path.join(local_directory, file_name)

    storage.child("docs/" + file_name).download(local_file_path)
    document_text = extract_text_from_pdf(local_file_path)

    if not document_text:
        print("No text found in the PDF file.")
        return "No text data found in PDF."

    sentences = sent_tokenize(document_text)
    tokenized_sentences = [preprocess_text(sentence) for sentence in sentences]
    model = None

    if model_type == 'word2vec':
        model = Word2Vec(tokenized_sentences, size=100, window=5, min_count=1, sg=1)
    elif model_type == 'glove':
        global glove_model
        model = glove_model
    else:
        print(f"Unsupported model type: {model_type}")
        return "Unsupported model type."
    if model is None:
        return "Model loading error."
    
    question_embedding = calculate_sentence_embedding(question, model)
    max_similarity = -1
    most_relevant_sentence = None

    for sentence in sentences:
        sentence_embedding = calculate_sentence_embedding(sentence, model)
        similarity = calculate_cosine_similarity(question_embedding, sentence_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = sentence

    answer = most_relevant_sentence.strip() if most_relevant_sentence else "No relevant answer found."
    # os.remove(local_file_path)
    print("ANSWER")
    print(model_type)
    print(answer)
    return answer

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def calculate_sentence_embedding(sentence, model):
    sentence_embedding = np.zeros(model.vector_size)
    num_words = 0
    tokens = preprocess_text(sentence)
    vector_model = model.wv if hasattr(model, 'wv') else model
    
    for word in tokens:
        if word in vector_model: 
            sentence_embedding += vector_model[word]  
            num_words += 1
    if num_words > 0:
        sentence_embedding /= num_words
    return sentence_embedding



def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

app = Flask(__name__)
app.secret_key = 'vaibhav290900'

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if request.form.get('btn') == 'index':
#             upload = request.files['upload']
#             global file_name
#             file_name = upload.filename
#             storage.child("docs/" + file_name).put(upload)
#             model_type = request.form.get('model_type', 'word2vec')
#             session['model_type'] = model_type
#             print(model_type)
#             return redirect(url_for('qa'))
#         elif request.form.get('btn') == 'qa':
#             question = request.form.get('question')
#             model_type = session.get('model_type', 'word2vec')
#             print("Model type in qa" + model_type)
#             answer = fine_tuning_drive(question, file_name, model_type)
#             return render_template('qa.html', answer=answer, question=question)
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('btn') == 'index':
            upload = request.files['upload']
            filename = upload.filename
            storage.child("docs/" + filename).put(upload)
            model_type = request.form.get('model_type', 'word2vec')
            session['model_type'] = model_type  # Store model type in session
            session['file_name'] = filename  # Store file name in session for reuse
            return redirect(url_for('qa'))
        elif request.form.get('btn') == 'qa':
            question = request.form.get('question')
            model_type = session.get('model_type', 'word2vec')  # Retrieve model type from session
            file_name = session.get('file_name')  # Retrieve file name from session
            answer = fine_tuning_drive(question, file_name, model_type)
            return render_template('qa.html', answer=answer, question=question, model_type=model_type)
    return render_template('index.html')



@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    global func
    func = request.args.get('type')
    return render_template('upload.html')

@app.route('/qa/', methods=['GET', 'POST'])
def qa():
    return render_template('qa.html')
	
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)