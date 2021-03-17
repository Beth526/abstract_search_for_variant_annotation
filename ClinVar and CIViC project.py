import requests
import re
import time

from bs4 import BeautifulSoup
import numpy as np
import pickle

import spacy
import scispacy
import en_ner_bionlp13cg_md

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#### Part 1: Getting abstracts for training data ####

def get_clinvar_ids():
    '''
    E-utililities https://www.ncbi.nlm.nih.gov/books/NBK25499/ and https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/
    '&retmax' is the max results to return 
    This returns all ids used as evidence in ClinVar
    '''
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=clinvar&term=[gene]&retmax=1000000'
    output = requests.get(url)
    page = output.text
    soup = BeautifulSoup(page, "lxml")
    ids = soup.find_all('id')
    clinvar_ids = []
    ids = soup.find_all('id')
    for element in ids:
        clinvar_ids.append(element.text)
    clinvar_ids = set(clinvar_ids)
    clinvar_ids = list(clinvar_ids)
    
    return clinvar_ids

def get_nonclinvar_ids(clinvar_ids):
	'''
	This returns 65828 random abstracts from 2018 that don't overlap the ClinVar ids
	'''

	url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=2018[pdate]&"has abstract"[Filt]&english[Filt]&human[Mesh]&retmax=65828&usehistory=y'
	output = requests.get(url)
	page = output.text
	soup = BeautifulSoup(page, "lxml")
	nonclinvar_pmids = []
	ids = soup.find_all('id')
	for element in ids:
	    nonclinvar_pmids.append(element.text)

	for i in set(nonclinvar_pmids).intersection(set(clinvar_pmids)):
    	nonclinvar_pmids.remove(i)

    return nonclinvar_pmids()

#civic pmids are directly imported from saved search results on civic

def get_noncivic_ids(civic_ids):
	'''
	This returns 10000 abstracts from 2018 that match the MESH terms Cancer, Protein or Gene
	They are checked for overlap to civic_pmids 
	'''

	url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=2018[pdat]+AND+("Neoplasms"[Mesh]+OR+"Proteins"[Mesh]+OR+"Genes"[Mesh])&retmax=10000&usehistory=y'
	output = requests.get(url)
	page = output.text
	soup = BeautifulSoup(page, "lxml")
	noncivic_pmids = []
	ids = soup.find_all('id')
	for element in ids:
	    noncivic_pmids.append(element.text)

	for i in set(noncivic_pmids).intersection(set(civic_pmids)):
    	noncivic_pmids.remove(i)

    return noncivic_pmids()


def convert_ids_to_lists(pmids):
	'''
	create lists of pmids to look up at once using NCBI's EUtiliies
	lookup 200 at once
	'''
	
	pmids_long=[]

	for i in np.arange(0,len(pmids)//200*200,200):
		j = i + 200
		temp = ','.join(pmids[i:j])
		pmids_long.append(temp)

	pmids_long.append(','.join(pmids[len(pmids)//200*200:]))

	return pmids_long


def get_abstracts_from_ids(pmids_long):
	'''
	get the abstracts matching the pmids
	this can be edited to also get journal, year, title, and pmid (see streamlit app)
	'''

	final_abstracts_list=[]

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
        
    
    return final_abstracts_list



clinvar_pmids = get_clinvar_ids()
nonclinvar_pmids = get_nonclinvar_ids(clinvar_pmids)

civic_pmids = pd.read_csv('CIViC_sources_2021-03-14T11_03_07.csv')
civic_pmids = civic_pmids['Citation ID'].unique()
noncivic_pmids = get_noncivic_ids(civic_pmids)

clinvar_abstracts = get_abstracts_from_ids(convert_ids_to_lists(clinvar_pmids))
nonclinvar_abstracts = get_abstracts_from_ids(convert_ids_to_lists(nonclinvar_pmids))

civic_abstracts = get_abstracts_from_ids(convert_ids_to_lists(civic_pmids))
noncivic_abstracts = get_abstracts_from_ids(convert_ids_to_lists(nonclivic_pmids))




#### Part 2: Tokenize the abstracts, remove gene names and gene products ####


def keep_token(t):
	'''
	remove the NER-labeled gene tokens to remove bias toward overepresented genes
	remove tokens that are only one character
	'''
    return (t.ent_type_ != 'GENE_OR_GENE_PRODUCT') and (t.__len__() > 1) 

def remove_gene_names_and_tokenize(abstracts_list, nlp):
	'''
	tokenize the abstracts into words
	lemmatize and lowercase 
	remove numbers
	'''
	docs = nlp.pipe(final_abstracts_list_non)
	token_docs = []

	for doc in docs:
	    token_docs.append([re.sub('[)(]|\-$|[0-9]*','',t.lemma_.lower()) for t in doc if keep_token(t)])
	    
	return token_docs

	
#get the tokens with scispacy
nlp = en_ner_bionlp13cg_md.load(disable=['tagger','parser'])

clinvar_abstracts = remove_gene_names_and_tokenize(clinvar_abstracts)

nonclinvar_abstracts = remove_gene_names_and_tokenize(nonclinvar_abstracts)

civic_abstracts = remove_gene_names_and_tokenize(civic_abstracts)

noncivic_abstracts = remove_gene_names_and_tokenize(noncivic_abstracts)


#convert tokens to numbers with keras tokenizer
tokenizer_clinvar = tf.keras.preprocessing.text.Tokenizer(num_words=20000)

tokenizer_clinvar.fit_on_texts(civic + noncivic)

clinvar_token = tokenizer_clinvar.texts_to_sequences(clinvar)

nonclinvar_token = tokenizer_clinvar.texts_to_sequences(nonclinvar)

word_index_clinvar = tokenizer_clinvar.word_index

with open('tokenizer_clinvar.pickle','wb') as write_file:
    pickle.dump(tokenizer_clinvar, write_file)


tokenizer_civic = tf.keras.preprocessing.text.Tokenizer(num_words=20000)

tokenizer_civic.fit_on_texts(civic + noncivic)

civic_token = tokenizer_civic.texts_to_sequences(civic)

noncivic_token = tokenizer_civic.texts_to_sequences(noncivic)

word_index_civic = tokenizer_civic.word_index

with open('tokenizer_civic.pickle','wb') as write_file:
    pickle.dump(tokenizer_civic, write_file)

#pad and cut the sequences off at lenght 250

clinvar_token = pad_sequences(clinvar_token, maxlen=250)

nonclinvar_token = pad_sequences(nonclinvar_token, maxlen=250)

civic_token = pad_sequences(civic_token, maxlen=250)

noncivic_token = pad_sequences(noncivic_token, maxlen=250)

#create a embedding matrix based on previous word2vec model trained on pubmed abstracts
#requires gensim to be installed

with open('w2v_model.pickle', 'rb') as read_file:
    w2v_model = pickle.load(read_file)

def create_embedding_matrix(word_index, w2v_model, w2v_size):
	'''
	The embedding matrix is used in the first set of the NN
	The emedding vectors from the w2v model that exist for words in the word index
	are placed in the order of token ids in the embedding matrix

	If a word wasn't in w2v model it would just be zeros for the embedding
	I think it might be better for these to be randomized so at least they could 
	be distinguisable? 
	'''
	embedding_matrix = np.zeros((nb_words, w2v_size))

	nb_words = len(word_index)+1
	for word, i in word_index.items():
	    if word in w2v_model:
	        embedding_matrix[i] = w2v_model[word]

	return embedding_matrix


embedding_matrix_clinvar = create_embedding_matrix(word_index_clinvar, w2v_model, 30)

embedding_matrix_civic = create_embedding_matrix(word_index_civic, w2v_model, 30)




##### Part 3: Train CIViC model #####

#train, val and test sets

civic_labels = [1]*len(civic_token)+[0]*len(noncivic_token)

civic_data = np.concatenate((civic_token, noncivic_token))

train_texts, test_texts, train_labels, test_labels = train_test_split(civic_data, civic_labels, test_size=.1, random_state=123, shuffle=True, stratify=labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.1, random_state=123, shuffle=True, stratify=train_labels)

class_weight = {0:1,1:(len(noncivic_token)/len(civic_token))}

#Define model

sequence_1_input = tf.keras.layers.Input(shape=(250,), dtype='int32')

embedding_layer = tf.keras.layers.Embedding(nb_words,
        30,
        weights=[embedding_matrix_civic],
        input_length=250,
        trainable=False)

embedded_sequence = embedding_layer(sequence_1_input)

X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
    embedded_sequence)
X = tf.keras.layers.GlobalMaxPool1D()(X)
X = tf.keras.layers.Dense(50, activation="relu")(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(1, activation="sigmoid")(X)

model = tf.keras.Model(inputs=[sequence_1_input], outputs=[X])

#initialize the model

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

model.summary()

#set early stopping rule

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#train (went for 32 epochs)
hist = model.fit(train_texts, train_labels, \
        validation_data=(val_texts, val_labels), \
        epochs=200, batch_size=50, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping])


tf.keras.models.save_model(model,'civic_model')

#evaluate perfomance

test_preds = model.predict(test_texts)

print(classification_report(test_labels,test_preds>0.5))




#### Part 4: Train ClinVar model ####

#train, val and test sets

cinvar_labels = [1]*len(clinvar_token)+[0]*len(nonclinvar_token)

clinvar_data = np.concatenate((clinvar_token, nonclinvar_token))

train_texts, test_texts, train_labels, test_labels = train_test_split(clinvar_data, civic_labels, test_size=.1, random_state=123, shuffle=True, stratify=labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.1, random_state=123, shuffle=True, stratify=train_labels)

#Define model

sequence_1_input = tf.keras.layers.Input(shape=(250,), dtype='int32')

embedding_layer = tf.keras.layers.Embedding(nb_words,
        30,
        weights=[embedding_matrix_clinvar],
        input_length=250,
        trainable=False)

embedded_sequence = embedding_layer(sequence_1_input)

X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(
    embedded_sequence)
X = tf.keras.layers.GlobalMaxPool1D()(X)
X = tf.keras.layers.Dense(50, activation="relu")(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(1, activation="sigmoid")(X)

model = tf.keras.Model(inputs=[sequence_1_input], outputs=[X])

#initialize the model

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

model.summary()

#set early stopping rule

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#train (went for 6 epochs)
hist = model.fit(train_texts, train_labels, \
        validation_data=(val_texts, val_labels), \
        epochs=200, batch_size=50, shuffle=True, \
        callbacks=[early_stopping])


tf.keras.models.save_model(model,'clinvar_model')

#evaluate perfomance

test_preds = model.predict(test_texts)

print(classification_report(test_labels,test_preds>0.5))




