import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import sklearn
import pickle
from ast import literal_eval
import data_clean as dc

x_test=pd.read_csv('x_test.csv',header=None).iloc[:,0]
y_test=pd.read_csv('y_test.csv',header=None).iloc[:,0]
x_train=pd.read_csv('x_train.csv',header=None).iloc[:,0]
y_train=pd.read_csv('y_train.csv',header=None).iloc[:,0]

#Logistic Regression Model
def logreg():
    df=pd.read_csv('logistic_regression_params.csv',index_col=None)
    row=df.iloc[df['f_score_weighted'].argmax(),:]
    ngram=literal_eval(row.ngram)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False, ngram_range=ngram, \
                       min_df=row.min_df,max_df=row.max_df, max_features=row.max_feature)
    x_train_tfidf=tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf=tfidf_vectorizer.transform(x_test)
    sc=sklearn.preprocessing.MaxAbsScaler()
    x_train_sc=sc.fit_transform(x_train_tfidf)
    x_test_sc=sc.transform(x_test_tfidf)

    clf=LogisticRegression(penalty='l1',C=row.C)
    clf=clf.fit(x_train_sc,y_train)
    y_test_pred = clf.predict(x_test_sc)
    score=sklearn.metrics.f1_score(y_test,y_test_pred, average='weighted')
    cm=confusion_matrix(y_test, y_test_pred)

    return clf, cm, tfidf_vectorizer, sc, score

if __name__=='__main__':
    #save logreg
    clf,cm,tfidf,sc,score=logreg()

    with open('logreg_model.pkl','wb') as f:
        pickle.dump(clf,f,protocol=pickle.HIGHEST_PROTOCOL)

    with open('logreg_tfidf.pkl','wb') as f:
        pickle.dump(tfidf,f,protocol=pickle.HIGHEST_PROTOCOL)

    with open('logreg_sc.pkl','wb') as f:
        pickle.dump(sc,f,protocol=pickle.HIGHEST_PROTOCOL)

    pd.DataFrame(cm,columns=['Group_1','Group_2','Group_3']).to_csv('logreg_cm.csv')

'''
    #test model
    with open('logreg_model.pkl', 'rb') as f:
        model=pickle.load(f)

    with open('logreg_tfidf.pkl', 'rb') as f:
        tfidf=pickle.load(f)

    with open('logreg_sc.pkl', 'rb') as f:
        sc=pickle.load(f)

    with open('test.txt','r') as f:
        text=f.read()

    text=dc.clean_words(text)
    text=[text]
    text_tfidf=tfidf.transform(text)
    text_sc=sc.transform(text_tfidf)

    pred=model.predict(text_sc)
    print(pred)
'''
