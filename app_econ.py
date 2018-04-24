
# coding: utf-8

# In[2]:


from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
from sklearn.feature_extraction.text import HashingVectorizer


# In[3]:


# loading in the pickled model
clf2 = pickle.load(open('logReg_model', 'rb'))


# In[5]:


# importing x_train data
X_train = pd.read_csv("/raw_data_econ/X_train_econ.csv", encoding="'iso-8859-1'")
X_train.head()


# In[7]:


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


# In[10]:


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


# In[11]:


# initialize flask 
app = Flask(__name__)    


# In[12]:


# render the homepage
@app.route("/")
def render_index():
     return render_template('index_econ.html')


# In[13]:


# When user submits text, returns prediction
@app.route("/", methods=["POST"])
def post_form():
    text = request.form['text']
    text_clean = clean_text(text)
    predict = model_predict(text_clean)
    return predict
    


# In[ ]:


# run app
if __name__=='__main__':
    app.run()
