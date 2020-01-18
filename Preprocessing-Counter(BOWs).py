# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 03:55:12 2020

@author: MMOHTASHIM
"""

import nltk
import argparse
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize 
import re
from tqdm import tqdm
import random 

parser=argparse.ArgumentParser(description="Creating a BOW through Indexation")
args=parser.parse_args()


def main():
    corpus=np.load("Annotations_corpus.npy",allow_pickle=True)
    
    stopword = nltk.corpus.stopwords.words('english') 
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    X_final_raw=[]
    for sentence in tqdm(corpus):
        for word in sentence:
            if len(word)>1:
                sentence=random.sample(sentence,1)[0]
                break
        sentence = sentence.lower()
        filtered_sentence = [w for w in word_tokenize(sentence) if not w in stopword]
        junk_free_sentence = []
        for word in filtered_sentence:
            junk_free_sentence.append(re.sub("[^\w\s]", " ", word))
        stemmed_sentence = []
        for w in junk_free_sentence:
            stemmed_sentence.append(stemmer.stem(w))
        
        lemmatized_sentence = []
        for w in stemmed_sentence:
            lemmatized_sentence.append(lemmatizer.lemmatize(w))
            
            
        X_final_raw.append(' '.join(lemmatized_sentence))
    lexicon_words=[]
    for sentence in tqdm(X_final_raw):
            for word in word_tokenize(sentence):
                lexicon_words.append(word)
    lexicon_words=list(set(lexicon_words)) 
    X_Counter=[]
    
    for sentence in tqdm(X_final_raw):
        instance=[0]*len(lexicon_words)
        for word in word_tokenize(sentence):
            index_word=lexicon_words.index(word)
            instance[index_word]=1
        X_Counter.append(instance)
        
        
    np.save("X_Counter.npy",np.array(X_Counter))


if __name__=="__main__":
    main()
           
    
    
    
    
    
    
    
    
            
    


