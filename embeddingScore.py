#%%
import pandas as pd 
import numpy as np 
from textblob import TextBlob
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from spellchecker import SpellChecker
from sklearn.metrics.pairwise import cosine_similarity

#%%
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')


class wordEmbedding:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.englishwords = set(nltk.corpus.words.words())
        self._token_to_idx = {}
        self._idx_to_token = {}
        self.spell = SpellChecker(distance=1)

    def cleanAndEmbed(self, df,columnName,removeNonEng=True,removeStopwords=True,removeLemma=True):
        sentenceEmbeddings = np.zeros(shape=(df.shape[0],200))
        for indexed,row in df.iterrows():
            sentences = sent_tokenize(row[str(columnName)])
            text = [re.sub('[^\sa-zA-Z]+',r'',text.lower()) for text in sentences]
            cleaned_words = []
            singleSentenceEmbeddings = np.zeros(shape=(200,))
            cleaned_words.extend([word_tokenize(sentence) for sentence in text])
            flatten = lambda outer_list: [item for inner_list in outer_list for item in inner_list]
            words = flatten(cleaned_words)
            words = [self.spell.correction(w) for w in words]
            if removeNonEng:
                words = [w for w in words if w in self.englishwords]
            if removeLemma:
                words= [self.lemmatizer.lemmatize(i) for i in words]
            if removeStopwords:
                words = [w for w in words if not w in self.stop_words]
            j = 1
            for token in words:
                if token not in self._token_to_idx:
                    index = len(self._token_to_idx)
                    self._token_to_idx[token] = index
                    self._idx_to_token[index] = token
                try: 
                    embeddings = self.model[token]
                except KeyError:
                    embeddings = np.zeros(shape=(200,))
                singleSentenceEmbeddings = ((j-1) * (singleSentenceEmbeddings) + embeddings) / j
                j+=1
            sentenceEmbeddings[indexed] =  singleSentenceEmbeddings
        return sentenceEmbeddings


    def loadGloveModel(self):
        print("loading glove model")
        file_path = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(file_path,'glove.twitter.27B.200d.txt'),'r',encoding='utf-8')
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordVectors = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordVectors
        print(len(gloveModel)," words loaded!")
        self.model = gloveModel

    # def CreateTweetEmbeddings(self):
    #     Emb_matrix = np.zeros((len(self._token_to_idx), 200))
    #     for i, word in enumerate(self._token_to_idx):
    #         try: 
    #             Emb_matrix[i] = self.model[word]
    #         except KeyError:
    #             Emb_matrix[i] = np.zeros(size=(200,))
    #     return Emb_matrix

#%%
if __name__ == "__main__":

    WE =  wordEmbedding()
    WE.loadGloveModel()

    safetyfile = open('safety.txt','r+')
    safetyfilelist = safetyfile.read().split(',')
    safetydf = pd.DataFrame(safetyfilelist, columns=['Text'])
    safetyEmbeddings = WE.cleanAndEmbed(safetydf,'Text')
    safetyEmbeddings = np.mean(safetyEmbeddings, axis=0)

    securityfile = open('security.txt','r+')
    securityfilelist = securityfile.read().split(',')
    securitydf = pd.DataFrame(securityfilelist, columns=['Text'])
    securityEmbeddings = WE.cleanAndEmbed(securitydf,'Text')
    securityEmbeddings = np.mean(securityEmbeddings, axis=0)
#%%
    df = pd.read_csv('Clerk.csv')
    print(df.columns)
    sentenceEmbeddings = WE.cleanAndEmbed(df,'Tweet_text')

    for index,_ in df.iterrows():
        df.loc[index,'SafeScore'] = np.linalg.norm(sentenceEmbeddings[index] - safetyEmbeddings)
        df.loc[index, 'SecureScore'] = np.linalg.norm(sentenceEmbeddings[index] - securityEmbeddings)

    df.to_csv('ClerkWSimi.csv')
# %%
