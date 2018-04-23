import pandas as pd
import numpy as np
import flask
from flask import request
import os
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import sklearn
from ast import literal_eval
import en_core_web_sm

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


if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
