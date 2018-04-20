import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import sklearn

x_test=pd.read_csv('x_test.csv',header=None).iloc[:,0]
y_test=pd.read_csv('y_test.csv',header=None).iloc[:,0]
x_train=pd.read_csv('x_train.csv',header=None).iloc[:,0]
y_train=pd.read_csv('y_train.csv',header=None).iloc[:,0]

#set params
ngrams=[(1,2),(1,3),(1,4)]
min_dfs=[5,10,20]
max_dfs=[0.1,0.2,0.3]
max_features=[50000,100000]
Cs=[0.1,1,5,10,20]


columns=['ngram','min_df','max_df','max_feature','C','f_score_weighted']
values=[[],[],[],[],[],[]]


i=1

for ngram in ngrams:
    for min_df in min_dfs:
        for max_df in max_dfs:
            for max_feature in max_features:
                for C in Cs:
                    print('this is '+str(i)+' round.')
                    i+=1
                    tfidf_vectorizer = TfidfVectorizer(analyzer='word', lowercase=False, ngram_range=ngram, \
                                       min_df=min_df,max_df=max_df, max_features=max_feature)

                    x_train_tfidf=tfidf_vectorizer.fit_transform(x_train)
                    x_test_tfidf=tfidf_vectorizer.transform(x_test)
                    sc=sklearn.preprocessing.MaxAbsScaler()
                    x_train_sc=sc.fit_transform(x_train_tfidf)
                    x_test_sc=sc.transform(x_test_tfidf)

                    clf=LogisticRegression(penalty='l1',C=C)
                    clf=clf.fit(x_train_sc,y_train)
                    y_test_pred = clf.predict(x_test_sc)
                    score=sklearn.metrics.f1_score(y_test,y_test_pred, average='weighted')

                    values[0].append(ngram)
                    values[1].append(min_df)
                    values[2].append(max_df)
                    values[3].append(max_feature)
                    values[4].append(C)
                    values[5].append(score)

dict={columns[i]:values[i] for i in range(len(columns))}
df=pd.DataFrame(dict,columns=columns).to_csv('logistic_regression_params.csv')
