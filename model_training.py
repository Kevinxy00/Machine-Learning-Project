import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightning.classification import CDClassifier

#word to vector
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
hash_vectorizer = HashingVectorizer(analyzer='word', ngram_range=(1, 1),n_features=50000)


#model training
def model_training(vect, clf):
    x_test=pd.read_csv('x_test.csv').iloc[:,0].tolist()
    y_test=pd.read_csv('y_test.csv').iloc[:,0].tolist()
    x_train=pd.read_csv('x_train.csv').iloc[:,0].tolist()
    y_train=pd.read_csv('y_train.csv').iloc[:,0].tolist()
    #x_sample=pd.read_csv('x_sample.csv').iloc[:,1].tolist()
    #y_sample=pd.read_csv('y_sample.csv').iloc[:,1].tolist()

    x_train=vect.fit_transform(x_train)
    x_test=vect.transform(x_test)
    clf=clf.fit(x_train,y_train)
    score=clf.score(x_test,y_test)
    return score


if __name__=='__main__':
    clf=LogisticRegression(penalty='l1', C=0.5)
    # clf=CDClassifier(penalty='l1')


    s=model_training(hash_vectorizer,clf)
    print(s)
