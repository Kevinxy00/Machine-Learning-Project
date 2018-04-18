import pandas as pd
import numpy as np
import spacy

nlp = spacy.load('en')
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

# split train test data
def train_test_split(dfs, pct):
    df_tests=[]
    df_trains=[]
    for df in dfs:
        sep=round(df.shape[0]*pct)
        df=df.sample(frac=1,random_state=42)
        df_tests.append(df.iloc[0:sep])
        df_trains.append(df.iloc[sep:])
    df_test=pd.concat(df_tests,ignore_index=True).sample(frac=1,random_state=42)
    df_train=pd.concat(df_trains,ignore_index=True).sample(frac=1,random_state=42)
    x_test=df_test['abstract'].apply(clean_words)
    y_test=df_test['label']
    x_train=df_train['abstract'].apply(clean_words)
    y_train=df_train['label']
    return x_test, y_test, x_train, y_train  # all in pandas dataframe form

if __name__=='__main__': #save splitted data into csv
    files=['group_0.csv','group_1.csv','group_2.csv']
    dfs=[pd.read_csv(i) for i in files]
    output=['x_test.csv','y_test.csv','x_train.csv', 'y_train.csv']
    i=0
    for d in train_test_split(dfs, 0.2):
        d.to_csv(output[i], index=False)
        i+=1
