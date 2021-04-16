#%%
# For Stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# basic operations
import re
import os
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# \logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


def sent_to_words(sentences):
    """ Used Genism's Simple Preprocess convert into words"""
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def small_preprocess(data):
    """We can put more small preprocesses here"""
   
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    return data
    

def small_preprocess_singlerow(data):
    """We can put more small preprocesses here"""
    # Remove new line characters
    data = re.sub('\s+', ' ', data) 
    # Remove distracting single quotes
    data = re.sub("\'", "", data)

    return data

def Chunking_BI_TRI(data):
    """ Creating Bigram and Trigram words"""
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return bigram_mod , trigram_mod

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def CoherenceAndPerplexity(lda_model,corpus,data,id2word):
    # Compute Perplexity
    perplexity = lda_model.log_perplexity(corpus)  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data, dictionary=id2word, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()

    return perplexity, coherence_score

def Visualize(lda_model,corpus,id2word):
    return pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i , row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df

def representative_topics(df):
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], axis=0)

    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    return sent_topics_sorteddf_mallet

def Topic_dist_Doc(df):
    # Number of Documents for Each Topic
    topic_counts = df['Dominant_Topic'].value_counts()

    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)

    # Topic Number and Keywords
    topic_num_keywords = df[['Dominant_Topic', 'Topic_Keywords']]

    # Concatenate Column wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    return df_dominant_topics

    
def topic_modelling_allrows_clubbed(df,column,mallet_needed=False):


    data = df[~pd.isna(df.loc[:,str(column)])][str(column)].values.tolist()

    
    # Perform basic Preprocess based on the dataset
    data = small_preprocess(data)

    #Create list of words from the sentences
    data_words = list(sent_to_words(data))

    #Creating Bigram Model
    bigram_mod,trigram_mod = Chunking_BI_TRI(data_words)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Topic modelling
    if mallet_needed :
        os.environ['MALLET_HOME'] = 'C:\\Users\\potta\\Documents\\RA\\BLM_Sentiment_analysis\\mallet-2.0.8\\bin\\mallet.bat'
        mallet_path = 'C:\\Users\\potta\\Documents\\RA\\BLM_Sentiment_analysis\\mallet-2.0.8\\bin\\mallet.bat'
        lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=2, id2word=id2word)

    else:
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    
    doc_lda = lda_model[corpus]

    # Deriving Perplexity and coherence score
    perplexity, coherence_score = CoherenceAndPerplexity(lda_model,corpus,data_lemmatized,id2word)

    #visualize the topics and most salient terms
    Visualize(lda_model,corpus,id2word)

    #Dominant topic and its score in the text
    sent_topics_df = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    # Group top 5 sentences under each topic
    df_representative_topic = representative_topics(sent_topics_df)

    return lda_model.print_topics(), perplexity, coherence_score , df_dominant_topic , df_representative_topic



def topic_modelling_single_row(df,column,index,mallet_needed):

    data = df.loc[index,str(column)]
    # Perform basic Preprocess based on the dataset
    data = small_preprocess_singlerow(data)

    #Create list of words from the sentences
    data_words = gensim.utils.simple_preprocess(str(data), deacc=True)

    #Creating Bigram Model
    bigram_mod, trigram_mod = Chunking_BI_TRI(data_words)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)


    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Topic modelling
    if mallet_needed :
        mallet_path = 'C:\\Users\\potta\\Documents\\RA\\BLM_Sentiment_analysis\\mallet-2.0.8\\bin\\mallet'
        lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=2, id2word=id2word)

    else:
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=1, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    

    return lda_model.print_topics()

#%%
if __name__ == "__main__":
    # Data_df = pd.read_csv('industry-purpose.csv',index_col=0)
    # for index in range(0,len(Data_df)):
    #     list1 = topic_modelling_single_row(Data_df,'Text description ',index,False)
    #     Data_df.loc[index,'Text Description Topic Modelling'] = str(list1[0][1])
    # Data_df.to_csv('industry-purpose.csv')

    Data_df = pd.read_csv('Forbes-Global-2000-List_mBucket_tm.csv',index_col=0)
    Data_df = Data_df[Data_df['BLM Mention\nAs of 21/09 list']=='YES']
    print(Data_df.columns)
    list1 , pp , cs, _ , _ = topic_modelling_allrows_clubbed(Data_df[:1000],'Text description ',False)
    print(list1)

# %%
