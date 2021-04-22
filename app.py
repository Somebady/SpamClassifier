# Spam Classifier
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from flask import Flask, render_template, request
# import jsonify
import requests
from os import path
import os
import pickle
import numpy as np
import pandas as pd
import sklearn
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
nltk.download('stopwords')
model = pickle.load(open('spam_classifier.pkl', 'rb'))
tv_vector = pickle.load(open('tv_vector.pkl', 'rb'))
@app.route('/', methods=['GET'])
def Home():
    return render_template('SpamClassifier.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        wordnet = WordNetLemmatizer()
        mail_details = []
        mail = request.form['mail']
        mail = re.sub('[^a-zA-Z]', ' ', mail)
        mail = mail.lower()
        mail = mail.split()
        mail_words = [wordnet.lemmatize(
            word) for word in mail if word not in set(stopwords.words('english'))]
        mail = ' '.join(mail_words)
        mail_details.append(mail)
        print(mail_details)
        mail_vector = tv_vector.transform(mail_details)
        print(mail_vector)
        print(mail_vector.shape)
        prediction = model.predict(mail_vector)
        print(prediction)
        if prediction[0] == 0:
            return render_template('SpamClassifier.html', prediction_texts="Geniune Mail !!")
        else:
            return render_template('SpamClassifier.html', prediction_texts="Alert!!, It's a spam, Be aware while opening it")
        #     # return render_template('flight_fare.html', prediction_text=f"Price of your ticket should be approx. {output}")
    else:
        return render_template('SpamClassifier.html')


if __name__ == "__main__":
    app.run(debug=True)
