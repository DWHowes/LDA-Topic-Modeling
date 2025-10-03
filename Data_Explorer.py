import spacy as sp
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

class DataExplorer():
    # Initialize the class. Load the spaCy model and the JSON file specified by filestr.
    def __init__(self, filestr:str) -> None:
        self.nlp = sp.load("en_core_web_lg")
        self.df = None
        self.word_list = self.__load_json(filestr)

## PRIVATE METHODS

    # Load the JSON file specified by filestr
    def __load_json(self, filestr:str)->list:
        self.df = pd.read_json(filestr)
        return self.df['paragraphs'] 
    
    def __create_corpus(self, text)->list:
        new = text.str.split()
        new = new.values.tolist()
        corpus = [word for i in new for word in i] 
        return corpus 

    # Return the number of n-grams specified by cnt. The type of n-gram returned (bigram, trigram)
    # is specified by the value of n.
    def __get_top_ngram(self, n:int, cnt:int):
        vec = CountVectorizer(ngram_range=(n, n)).fit(self.get_word_list())
        bag_of_words = vec.transform(self.get_word_list())
        sum_words = bag_of_words.sum(axis=0) 

        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

        return words_freq[:cnt]    
    
# PUBLIC METHODS

    # Return the word list
    def get_word_list(self)->list:
        return self.word_list
    
    def show_char_word(self, bincnt=10)->None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))

        char_map = self.word_list.str.len()
        word_map = self.word_list.str.split().map(lambda x: len(x))

        plt.subplot(1,2,1)
        plt.xlabel("Characters")
        plt.ylabel("Count")
        ax1.hist(char_map, bincnt)

        plt.subplot(1,2,2)
        plt.xlabel("Words")
        plt.ylabel("Count")
        ax2.hist(word_map, bincnt)

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png')
        st.image(buf)
    
    def show_common_words(self, cnt=40)->None:
        plt.subplots(figsize=(7,8))

        corpus = self.__create_corpus(self.get_word_list())

        counter = Counter(corpus)
        most = counter.most_common()

        x, y=[], []

        for word,count in most[:cnt]:
            if (word):
                x.append(word)
                y.append(count)
            
        fig = plt.figure(figsize=(8,7))
        g = sns.barplot(x=y,y=x, width=0.6)
        g.set(xlabel="Count", ylabel="Common Words", title="Common Words")
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        st.image(buf)

    def show_ngrams(self, n=2, cnt=10)->None:
        fig = plt.figure(figsize=(8,7))

        top_n_grams=self.__get_top_ngram(n, cnt)[:cnt]

        x, y = map(list,zip(*top_n_grams))
        g = sns.barplot(x=y,y=x)
        title_str = "N-Gram " + "(" + str(n) + ")"
        g.set(xlabel="Count", title=title_str)

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png')
        st.image(buf)

    def show_entity(self, entity:str, cnt=10)->None:
        nlp = self.nlp
        text = self.get_word_list()
        entity = entity.upper()
        fig = plt.figure(figsize=(8,7))

        def _get_ner(text,ent):
            doc=nlp(text)
            return [X.text for X in doc.ents if X.label_ == ent]

        entity_filtered=text.apply(lambda x: _get_ner(x,entity))
        entity_filtered=[i for x in entity_filtered for i in x]
        
        counter=Counter(entity_filtered)
        x,y=map(list,zip(*counter.most_common(cnt)))
        sns.barplot(x=y,y=x).set_title(entity)    

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png')
        st.image(buf)

