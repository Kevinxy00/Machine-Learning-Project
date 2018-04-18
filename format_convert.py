import pandas as pd

#get attributes from individual publication record
def get_pmid(pub):
    pmid=pub.split('\n')[0].strip()
    return pmid

def get_attr(pub, sep):
    '''
    get title or abstract
    '''
    sep=sep+r'  - '
    if sep in pub:
        attr_split=pub.split(sep)[1]\
                      .split(r' - ')[0]\
                      .split('\n')[0:-1]
        attr_split=[i.strip() for i in attr_split]
        attr=' '.join(attr_split)
        return attr
    else:
        return

def get_institute(pub):
    '''
    This is optional. Only need last author's institute.
    '''
    sep=r'AD  -'
    if sep in pub:
        attr_split=pub.split(sep)[-1]\
                      .split('.')[0]\
                      .split('\n')
        attr_split=[i.strip() for i in attr_split]
        attr=' '.join(attr_split)
        #remove zipcode and country
        attr_list=attr.split(';')
        attrs=[]
        for i in attr_list:
            i=i.split(',')[0:-2]
            i=' '.join(i)
            attrs.append(i)
        attr=';'.join(attrs)
        #remove uninformative words
        black_list=['school','university', 'department','of', 'the', 'institute', 'division',
        'center','centre','research','medical','diseases', 'national', 'international','bureau',
        ]
        attr=[i for i in attr.split() if i.lower() not in black_list]
        attr=' '.join(attr)
        return attr
    else:
        return

#parse medline and convert to csv
def medline_to_csv(f_name, j_name='test'):
    '''
    convert medline file into dataframe
    '''
    with open(f_name, 'r') as f:
        pubs=f.read().split('PMID-')
        pubs=[i for i in pubs if i!='\n']
        columns=['pmid','title','abstract','institute']
        values=[[],[],[],[]]
        for pub in pubs:
            values[0].append(get_pmid(pub))
            values[1].append(get_attr(pub,'TI'))
            values[2].append(get_attr(pub,'AB'))
            values[3].append(get_institute(pub)) #optional
        df_dict={columns[i]:values[i] for i in range(len(columns))}
        df=pd.DataFrame(df_dict)
        df['journal']=j_name
        df=df.loc[:,['pmid','title','abstract','institute','journal']]
    return df

if __name__=='__main__':
    test_file='sample.txt'
    print(medline_to_csv(test_file).head(10))
