#This is an Streamlit app

import streamlit as st
import pickle
from bs4 import BeautifulSoup
import requests
import tensorflow as tf
import numpy as np
from spacy.lang.en import English
import re
import pandas as pd
import base64


#import word2vec model (embedding size = 30)
with open('/data/data/w2v_model.pickle', 'rb') as read_file:
    w2V_model = pickle.load(read_file)

#import the civic tokenizer
with open('/data/data/tokenizer_civic.pickle', 'rb') as read_file:
    tokenizer_civic = pickle.load(read_file)

#import the clinvar tokenizer
with open('/data/data/tokenizer_clinvar.pickle', 'rb') as read_file:
    tokenizer_clinvar = pickle.load(read_file)

#import the civic model
civic_model = tf.keras.models.load_model('/data/data/civic_model')

#import the clinvar model
clinvar_model = tf.keras.models.load_model('/data/data/clinvar_model')

#define NCBI EUtilities Query function
def pubmed_fetch(*args):
    st.write('Fetching abstracts from Pubmed (~30 sec depending on number of results)')
    
    base='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term='
    for i in args:
        print(i)
    
    if len(list(args)) == 1:
        query = args
    if len(args) >= 1:
        query=''
        for i in range(len(args) - 1):
            query += args[i] + "+"
        query += args[-1]
    
    end = '&retmax=5000&usehistory=y'
    
    url = base + query + end
    output = requests.get(url)
    page = output.text
    soup = BeautifulSoup(page, "lxml")
    
    pmids = []
    ids = soup.find_all('id')
    for element in ids:
        pmids.append(element.text)
        
    pmids_long=[]
    for i in np.arange(0,len(pmids)//200*200,200):
        j=i+200
        temp = ','.join(pmids[i:j])
        pmids_long.append(temp)
    pmids_long.append(','.join(pmids[len(pmids)//200*200:]))
    
    final_abstracts_list=[]
    final_journals_list=[]
    final_pubmed_list=[]
    final_title_list=[]
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='

    for i in pmids_long:
        query = i
        url = base + query +'&retmax=1000'+'&usehistory=y'+'&retmode=xml'
        output = requests.get(url)
        page = output.text
        soup = BeautifulSoup(page, "lxml")
        
        test = soup.find_all('abstract')
        final_test=[]
        for i in test:
            temp = i.find_all('abstracttext')
            temp3=[]
            for j in temp:
                temp2 = j.text
                temp3.append(temp2)
            temp3 = ' '.join(temp3)

            final_test.append(temp3)
            
        final_abstracts_list.extend(final_test)
        
        for i in test: #error rarely if a book
            try:
                temp = i.findPrevious('isoabbreviation').text
                final_journals_list.append(temp)
                
            except:
                final_journals_list.append('NA')
                
        for i in test:
            temp = i.findPrevious('pmid').text
            final_pubmed_list.append(temp)
            
        for i in test:
            temp = i.findPrevious('articletitle').text
            final_title_list.append(temp)
    
    return final_abstracts_list, final_journals_list, final_title_list, final_pubmed_list
      
#define tokenization function
def tokenize_abstracts(final_abstracts_list, model):
    st.write('Tokenizing')
    
    nlp = English()
    spacy_tokenizer = nlp.tokenizer
    docs = nlp.pipe(final_abstracts_list)
    token_abs = []
    for doc in docs:
        token_abs.append([re.sub('[)(]|\-$|[0-9]*','',t.lemma_.lower()) for t in doc if (t.__len__() > 1)])
    
    if model == "CIViC":
        token_abs = tokenizer_civic.texts_to_sequences(token_abs)
        
    if model == "ClinVar":
        token_abs = tokenizer_clinvar.texts_to_sequences(token_abs)
        
    token_abs = tf.keras.preprocessing.sequence.pad_sequences(token_abs, maxlen=250)
    
    return token_abs

#define prediction table function
def score_abstracts(token_abs, final_journals_list, final_title_list, final_pubmed_list, final_abstracts_list, model):
    st.write('Scoring')
    
    if model == "CIViC":
        preds = civic_model.predict(token_abs)
        
    if model == "ClinVar":
        preds = clinvar_model.predict(token_abs)
        
    preds = [i[0] for i in preds]
    
    df = pd.DataFrame({'journal':final_journals_list,'title':final_title_list,'abstract':final_abstracts_list,'pmid':final_pubmed_list,'score':preds})
    
    df.sort_values('score', ascending=False, inplace=True)
    
    return df

#define download link function (from streamlit forum)
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download full results table with abstracts (csv file)</a>'
    return href
    

#First section on app info
st.write('''# ClinVar and CIViC Optimized Pubmed Search App
''')

st.write('''This is a tool for finding literature-based evidence for varient classifications. It searches pubmed for the input search terms, then takes the top 5,000 results and scores them based on similarity to existing ClinVar or CIViC evidence abstracts. The top scoring results are returned, and the full results table can be downloaded.

  - ClinVar: Focus on pathogenicity of germline mutations
  - CIViC: Focus on prognostic value and response to therapy for somatic mutations in cancer

This app uses a small word2vec model and bidrectional LSTM neural network to score the abstracts based on thier content. For ClinVar mode, it was trained on all ClinVar abstracts in Feb 2021 (~70k) vs an equal amount of random non-overlapping abstracts from 2017-2018. For CIViC mode, it was trained on all evidence abstracts from CIViC found in Feb 2021 (~3k) vs 10k non-overlapping abstracts that matched MESH terms Cancer, Protein or Gene from 2018. Before training, Scispacy NER was used to remove tokens tagged as a gene or gene product, in an attempt to reduce bias toward genes.
''')

#drop down menu and user input
model = st.selectbox(
    'Model', ['ClinVar']+
     sorted(['CIViC']) )

user_input = st.text_input("search term (example: a gene or disease name, can be multi-word)", '')

#run the analysis
if user_input != "":
    final_abstracts_list, final_journals_list, final_title_list, final_pubmed_list = pubmed_fetch(user_input)

    token_abs = tokenize_abstracts(final_abstracts_list, model)

    df = score_abstracts(token_abs, final_journals_list, final_title_list, final_pubmed_list, final_abstracts_list, model)

    #link to save the final dataframe

    st.markdown(get_table_download_link(df), unsafe_allow_html=True)

    #print top 50
    st.table(df.loc[:,df.columns != 'abstract'].head(50))
















