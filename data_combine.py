import pandas as pd
import format_convert as fc
import glob
import matplotlib.pyplot as plt
import numpy as np

def combine_pubs(path):
    '''
    Convert all medline files in path folder, and concat them into one dataframe.
    '''
    df_list=[]
    file_names=glob.glob(path+'*')
    for f_name in file_names:
        j_name=f_name.split('/')[-1].split('.')[0]
        df=fc.medline_to_csv(f_name, j_name=j_name)
        df_list.append(df)
    combined_df=pd.concat(df_list)
    #drop incorrect data
    combined_df['check']= (combined_df['abstract'].str.len() > 500)
    combined_df=combined_df.loc[combined_df['check'],:]
    combined_df['abstract']=combined_df['abstract']+combined_df['institute']
    combined_df=combined_df.loc[:,['pmid','title','abstract','journal']]
    combined_df.dropna(inplace=True)
    return combined_df

def get_stat(df):
    '''
    Get total records number, and count data in each journal.
    '''
    total=df.shape[0]
    df['count']=0
    df_count=df.groupby('journal').agg({'count':'count'}).reset_index()
    return total, df_count

def make_autopct(values): # assign values to function my_autopct
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct #return a function with values set

def make_pie(labels,counts,name):
    plt.style.use('seaborn')
    fig,ax=plt.subplots(figsize=(7,7))
    ax.pie(counts, labels=labels, autopct=make_autopct(counts))
    ax.set_title('Total Abstracts in Each Group')
    ax.legend()
    fig.savefig(name)
    fig.clf()
    return

def make_bar(df,title,name):
    plt.style.use('seaborn')
    x,y=np.arange(df.shape[0]),df['count'].values
    fig, ax=plt.subplots(figsize=(7,5))
    ax.bar(x,y)
    plt.xticks(x, df['journal'].values,rotation=45)
    ax.set_title(title)
    ax.set_xlabel('Journals')
    ax.set_ylabel('Number of Counts')
    plt.tight_layout()
    fig.savefig(name)
    fig.clf()
    return

#combine top journals
if __name__=='__main__':
    groups=['0','1','2']
    counts=[]
    df_counts=[]
    i=0
    for group in groups:
        path='raw_data/'+'group_'+group+'/'
        file_name=group+'.csv'
        combined=combine_pubs(path)
        combined['label']=i
        i+=1
        combined.to_csv(file_name)
        count,df_count=get_stat(combined)
        counts.append(count)
        df_counts.append(df_count)

    labels=['Group 0','Group 1', 'Group 2']
    make_pie(labels,counts,'total_count.jpg')

    titles=['Counts of Each Journal in '+i for i in labels]
    names=[i+'_counts'+'.jpg' for i in groups]
    for i in range(3):
        make_bar(df_counts[i],titles[i],names[i])
