import pandas as pd
import numpy as np
import flask
from flask import Flask, render_template, request
import os
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import sklearn
from ast import literal_eval
import en_core_web_sm

# ******  From app_econ.ipynb  ******
# loading in the pickled model
clf2 = pickle.load(open('logReg_model', 'rb'))

# importing x_train data
X_train = pd.read_csv("raw_data_econ/X_train_econ.csv", encoding="'iso-8859-1'")
X_train.head()

# function to clean text 
stop_words = set(stopwords.words('english'))
def clean_text(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    string = ' '.join(filtered_sentence)
    return string

# Function to predict if an input string is likely to be in top journal
# note: copy/pasted from Econ_machineLearn.ipynb
hash_vectorizer = HashingVectorizer(analyzer='word', ngram_range=(1, 2))
hash_vectorizer.fit(X_train)
def model_predict(s):
    string = []
    string.append(s)
    vectorized = hash_vectorizer.transform(string)
    probab = round(max(clf2.predict_proba(vectorized)[0])* 100, 2) 
    prediction = clf2.predict(vectorized)[0]
    if prediction == 1:
        result = "Predicted to be in the top 20 Economics journals"
    else:
        result = "Predicted to NOT be in the top 20 Economics journals"
    return result + " with a probability of " + str(probab) + "%."

# **** END app_econ.ipynb **** 

app = flask.Flask(__name__)

nlp= en_core_web_sm.load()
#nlp=spacy.load('en')

blacklist=['the','study','ref','here','role','to',
          '-PRON-','this','that','background','introduction','method','conclusion',
           'find']

#clean the input text
def clean_words(s):
    doc=nlp(s)
    str_list=[]
    for token in doc:
        if (not token.is_stop)&(token.pos_!='VERB')&(token.pos_!='ADP')&(token.lemma_ not in blacklist):
            str_list.append(token.lemma_)
    string=' '.join(str_list)
    return string

#load model
with open('logreg_model.pkl', 'rb') as f:
    model=pickle.load(f)

with open('logreg_sc.pkl', 'rb') as f:
    sc=pickle.load(f)

with open('logreg_tfidf.pkl', 'rb') as f:
    tfidf_f=pickle.load(f)

#df1=pd.read_csv('x_train_1.csv',header=None)
#df2=pd.read_csv('x_train_2.csv',header=None)
#df3=pd.read_csv('x_train_3.csv',header=None)
#x_train=pd.concat([df1,df2,df3]).iloc[:,0]

#tfidf = TfidfVectorizer(analyzer='word', lowercase=False, ngram_range=(1,4), \
#                   min_df=10,max_df=0.3, max_features=50000)
#tfidf_f=tfidf.fit(x_train)

@app.route('/')
def homepage():
     return flask.render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    text=clean_words(text)
    text=[text]
    text_tfidf=tfidf_f.transform(text)
    text_sc=sc.transform(text_tfidf)
    pred=model.predict(text_sc)
    if pred[0]==0:
        impact='above 27.'
    elif pred[0]==1:
        impact='between 10 and 27.'
    else:
        impact='below 10.'
    return 'I think this abstract is from the journal with impact factor '+impact

# Route for Econ Journal Abstract Predictor 
# note: adapted from app_econ.ipynb
# render the homepage
@app.route("/predict-econ")
def render_index():
     return render_template('index_econ.html')

# When user submits text, returns prediction
@app.route("/", methods=["POST"])
def post_form():
    text = request.form['text']
    text_clean = clean_text(text)
    predict = model_predict(text_clean)
    return predict
   
# **** END Route for Econ Journal Abstract Predictor ****

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
