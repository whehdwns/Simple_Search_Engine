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
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

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
        word = [lemmatizer.lemmatize(w) for w in word if not w in stopwords]
        word = " ".join(word)
        result.append(word)
    return result

def calculate_terms(listed):
    normalized_text = []
    collection_frequency = Counter()
    document_frequency = Counter()
    output_wordlist_dict ={}
    terms_frequency = defaultdict(lambda: Counter([]))
    
    normalized_text = preprocess(listed)
    
    for i in range(len(normalized_text)):
        tokenized_list= []
        for j in normalized_text[i].split():
            tokenized_list.append(j)
        output_wordlist_dict[i] = Counter(tokenized_list)
        collection_frequency.update(tokenized_list)
        document_frequency.update(set(tokenized_list))
        
    for key, value in output_wordlist_dict.items():
        for term, term_cnt in value.items():
            terms_frequency[term][key] += term_cnt
    
    return normalized_text, collection_frequency, document_frequency, output_wordlist_dict, terms_frequency

def dictionary_list(listed):
    sort_dict = {}
    result_sort_dict = {}
    offset_sum = 0
    offset_i = 0
    sort_dict = OrderedDict(sorted(listed.items()))
    for i, value in enumerate(sort_dict.keys()):
        offset_i = len(sort_dict[value]) * 2 
        result_sort_dict[value] = len(sort_dict[value].values()),offset_sum
        offset_sum = offset_sum + offset_i 
    return result_sort_dict

def idf_corpus(dict_corpus,N_corpus):
    idf_dict = {}
    for key_i in dict_corpus.keys():
        tf_i = dict_corpus.get(key_i)[0]
        idf_i = math.log2(N_corpus/tf_i)
        idf_dict[key_i] = idf_i
    return idf_dict

def tf_idf(post_list,idf_matrix):
    weight_matrix =[]
    for i, j in post_list.items():
        idf ={}
        for k in j:
            if k not in idf_matrix:
                idf_matrix[k] = 0
            else:
                idf[k] = idf_matrix[k]*j[k]
        weight_matrix.append(idf)                       
    return weight_matrix

def vector_length(weight):
    length_matrix = {}
    for doc_i in range(len(weight)):
        length_matrx = []
        for i in weight[doc_i].values():
            length_matrx.append(i)
        sum_of_squares = sum(map(lambda k : k * k, length_matrx))
        vlength = math.sqrt(sum_of_squares)
        length_matrix[doc_i] = vlength
    return length_matrix

def cosine_similarities(doc_weight, query_weight, doc_length, query_length, query_term_freq):
    N = len(doc_weight)
    cos_score = []
    for i in range(len(query_term_freq)):
        cos_score.append([0]*N)
        for j  in query_term_freq[i].keys():
            query_tfidf = 0
            if query_weight[i].get(j):
                query_tfidf = query_weight[i].get(j)
            for k in range(len(doc_weight)):
                if(query_length[i] != 0) & (doc_length[k] != 0):
                    if(doc_weight[k].get(j)):
                        #Document Length * Query Length
                        doc_query_length = doc_length[k] * query_length[i]
                        # tf-idf weight of term in document * tf-idf weight of term in query
                        doc_query_vector = doc_weight[k].get(j) * query_tfidf
                        cos_score[i][k] += doc_query_vector / doc_query_length  
    return cos_score


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

book_idf_matrix = idf_corpus(book_dict_pos_output,len(book_normalized_text))
book_weight = tf_idf(book_term_freq, book_idf_matrix)
book_length = vector_length(book_weight)

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
# Bible Search Engine
""")

# search bar
query = st.text_input("Search!", "")

num_search = st.slider('How many Result', 0, 10, 1)

query_content = []
query_content.append(query)

q_normalized_text, q_collec_freq, q_doc_freq, q_term_freq, q_posting_list_output = calculate_terms(query_content)
q_dict_pos_output = dictionary_list(q_posting_list_output)
q_idf_matrix = idf_corpus(q_dict_pos_output,len(q_normalized_text))
q_weight = tf_idf(q_term_freq, book_idf_matrix)
q_length = vector_length(q_weight)
cos_score_q = cosine_similarities(book_weight, q_weight, book_length, q_length, q_term_freq)

ranking_list =[]
for i in range(len(cos_score_q)):
    ranked = []
    for j in range(len(cos_score_q[i])):
        ranked.append([j, cos_score_q[i][j]])
    ranking_list.append(ranked)

for i in ranking_list:
    i.sort(key=lambda x: x[1], reverse=True)

every_result =[]
chapter_result=[]
score_result =[]
for i in ranking_list:
    result =[]
    for j in i[:num_search]:
        chapter_result.append(j[0])
        result.append(book_content[j[0]])
        score_result.append(j[1])
    every_result.append(result)

if len(query)!=0:
    if sum(score_result)==0:
        st.write(str(query)+' is not found')
    else:
        for i in range(len(every_result[0])):  
            st.write(str(bible_chapter_list[chapter_result[i]])+ ' - ' + str(every_result[0][i]))