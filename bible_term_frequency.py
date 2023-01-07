import streamlit as st

import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter, defaultdict, OrderedDict
import re
import sys
import os
import math
import string
import time
import operator
from itertools import islice
nltk.download('stopwords')
import pickle
import warnings
warnings.filterwarnings("ignore")

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
ps =PorterStemmer()

def normalization(word):
    word= word.lower()
    word = word.replace('\n','')
    word  = re.sub(r'[^\w\s]', '', word)
    word = re.sub('[0-9]', '', word)
    return word

def preprocess(data):
    result = []
    for line in data:
        word = normalization(line)
        word = word.lower().strip().split()
        stopwords = nltk.corpus.stopwords.words("english")
        word = [lemmatizer.lemmatize(lemmatizer.lemmatize(w , 'v'), 'a') for w in word if not w in stopwords]
        word = " ".join(word)
        result.append(word)
    return result

with open('pickle_set/book_norm_text.pickle', 'rb') as f:
    book_normalized_text = pickle.load(f)

with open('pickle_set/book_collec_freq.pickle', 'rb') as f:
    book_collec_freq = pickle.load(f)

with open('pickle_set/book_doc_freq.pickle', 'rb') as f:
    book_doc_freq = pickle.load(f)
    
with open('pickle_set/book_term_freq.pickle', 'rb') as f:
    book_term_freq = pickle.load(f)

with open('pickle_set/book_dict_pos_output.pickle', 'rb') as f:
    book_dict_pos_output = pickle.load(f)

with open('pickle_set/bible_chapter_list.pickle', 'rb') as f:
    bible_chapter_list = pickle.load(f)


book_content_w_ch =[]
with open('kjv.txt') as f:
    next(f)
    for line in f:
        book_content_w_ch.append(line)

book_content =[]
for i in book_content_w_ch:
    book_content.append(i.replace(i.split(' ')[0], ''))

# title and description
st.write("""
# Bible Terms Frequency
""")

# search bar
query = st.text_input("Search!", "")

query_content = []
query_content.append(query)
search_q = preprocess(query_content)

for i in search_q:
    term_freq = book_doc_freq[i]

term_freq_list=[]
for i in book_term_freq:
    if search_q[0] in book_term_freq[i].keys():
        term_freq_list.append(i)

num_search = st.slider('Limit Result', 0, len(term_freq_list), 1)


if len(query)!=0:
    if term_freq==0:
        st.write(str(query)+' is not found')
    else:
        st.write(str(query)+ ' apear '+ str(term_freq) +' times.')
        for i in term_freq_list[:num_search]:
            with st.expander(str(bible_chapter_list[i])):
                st.write(str(book_content[i]))