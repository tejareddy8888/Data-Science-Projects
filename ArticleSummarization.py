#%%
from wordcloud import WordCloud, STOPWORDS 
import json
import pandas as pd
from datetime import datetime, timedelta
from functools import reduce
from math import ceil
from os import path
from time import sleep
from bs4 import BeautifulSoup
from requests import get, codes
from requests_oauthlib import OAuth1
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
#from summarizer import Summarizer
from transformers import AutoModel,AutoModelWithLMHead, AutoTokenizer,pipeline
import gensim
import gensim.corpora as corpora
from gensim import corpora
from collections import defaultdict
from gensim import models
from gensim import similarities
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
import string
import nltk.data
import pytesseract
import PyPDF2


ua = UserAgent()

stop_words = set(stopwords.words('english')) 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not","didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is","I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would","i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would","it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam","mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have","mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have","she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is","should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as","this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would","there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have","they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have","wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are","we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are","what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is","where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have","why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have","would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have","you're": "you are", "you've": "you have"}

def FetchArticle(article_url):
    article_text = []
    url = article_url
    opts = Options()
    opts.add_argument("user-agent="+str(ua.chrome))
    driver = webdriver.Chrome(chrome_options=opts)
    driver.get(url)
    driver.implicitly_wait(20)
    driver.get_screenshot_as_file(str('screenshot')+'.png')
    elements = driver.find_elements_by_xpath('//p')
    for element in elements:
        article_text.append(element.text)  
    driver.close()
    return article_text

def clean_text(text):
    stop_words = set(STOPWORDS)
    alphabets = set(string.ascii_lowercase)
    text = text.lower()
    #after tweet preprocessing the colon symbol left remain
    text = re.sub(r'‚Ä¶', '', text)
    #replace the URLs
    text = re.sub(r'http[s]?[:]?[\s]?\/\/(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',' ',text)
    #replace the n't
    text = re.sub(r'n\'t', ' not', text)
    #replace the 's'
    text = re.sub(r'\'s', '', text)
    #replace the numbers
    text = re.sub(r'\d', '', text)
    #replace the dots
    text = re.sub(r'…', '', text)

    exclude = set(string.punctuation).union(set(['’','…','``','...','“','‘','”','\'','\n','\t','\'\'']))

    word_tokens = word_tokenize(text)
    #filter using NLTK library append it to a string
    filtered_text = [w for w in word_tokens]
    filtered_text = []
    #looping through conditions
    for w in word_tokens:
    #check tokens against stop words and punctuations
        if w not in exclude and w not in ('rt','…',) and w not in stop_words and w not in alphabets:
            filtered_text.append(w)
    return ' '.join(filtered_text)


def topicmodelling(text):
    text = clean_text(text)
    #print(text)
    texts = [word_tokenize(i) for i in text.split()]
    #print(texts)
    frequency = defaultdict(int)
    for token in texts:
        frequency[token[0]] += 1
    #print(frequency)
    texts = [token for token in texts if frequency[token[0]] >= 3] 
    #print(texts)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,
                                           num_topics=3, 
                                           random_state=42,
                                           update_every=0,
                                           chunksize=300,
                                           passes=20,
                                           alpha='auto',
                                           per_word_topics=True)

    return lda_model.print_topics(num_words=5)

def text_cleaner(text):
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() ]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()


basic_df = pd.read_csv('Companies_BLM_manualBucketed.csv',encoding='cp437')

for index , row in basic_df.iterrows():

    try:
        topics = topicmodelling(basic_df.loc[index,'Article_text']) 
        basic_df.loc[index,'Topics_Articles'] = '\n'.join([topics[0][1],topics[1][1],topics[2][1]])
    except Exception as e:
        print(str(e) + 'at '+str(index))

basic_df.to_csv('Companies_BLM_manualBucketed_w_topicmodel.csv')

""" Below Code is for Summarization and topic modelling"""
#%%

# for index , row in basic_df.iterrows():
#     fetchedarticle = FetchArticle(basic_df.loc[index,article_num])
#     fetchedarticle = list(filter(None, fetchedarticle))
#     uncleanArticle = ' '.join(fetchedarticle)
#     CleanedArticle = text_cleaner(uncleanArticle)
#     basic_df.loc[index,article_num+'_cleaned_text'] = CleanedArticle
#     basic_df.loc[index,article_num+'_uncleaned_text'] = uncleanArticle

# basic_df.to_csv('Onlyarticles_text.csv')

# #%%

# model = AutoModelWithLMHead.from_pretrained("bart-large-xsum")
# tokenizer = AutoTokenizer.from_pretrained("bart-large-xsum")
# summarizer = pipeline("summarization")

# to_csv = 0
# i=0

# date_dict = dict(basic_df[6:])

# for index , row in basic_df.iterrows():
#     for article_num in list(date_dict.keys())[6:]: 
#         if pd.notna(basic_df.loc[index,article_num]):
#             inputs = tokenizer.encode("summarize: " + basic_df.loc[index,article_num][:6000], return_tensors="pt", max_length=100000)
#             model.resize_token_embeddings(len(tokenizer))
#             outputs = model.generate(inputs, max_length=350, min_length=40,use_first = False)
#             basic_df.loc[index,'abstractive summary of'+article_num] = tokenizer.decode(outputs[0],skip_special_tokens=True, clean_up_tokenization_spaces=False)
#             basic_df.loc[index,'extractive summary of '+article_num] = summarizer(basic_df.loc[index,article_num], max_length=350, min_length=30)[0]['summary_text']
#             topics = topicmodelling(basic_df.loc[index,article_num])
#             basic_df.loc[index,'Topics of'+article_num] = topics[0][1]
#             basic_df.loc[index,'Topic2 of'+article_num] = topics[1][1]
#         to_csv+=1

# print('count of bagged blogs is'+str(to_csv))
# print('count of missed blogs is'+str(i))

# basic_df.to_csv('Summaries_Topics_BLM_137.csv')
# # %%

# # #%%
# # from summarizer import Summarizer
# # from summarizer.coreference_handler import CoreferenceHandler

# # handler = CoreferenceHandler(greedyness=.4)
# # model = Summarizer(sentence_handler=handler)

# # #%%
# # text = model(uncleanArticle[0:500],max_length =200, use_first = False)


# # # %%

# # from summarizer import Summarizer

# # BESmodel = Summarizer()
# # BESmodel(uncleanArticle,max_length =200, use_first = False)


# %%
