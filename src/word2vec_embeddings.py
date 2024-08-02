import pickle as pkl
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords # Not supporting Polish :/
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer # Not supporting Polish :/
from pystempel import Stemmer
import numpy as np
import re

class Word2VecEmbeddings:
    def __init__(self, model_path="", stopwords_path = "../data/polish.stopwords.txt"):

        with open(model_path, 'rb') as f:
            self.model = pkl.load(f)

        with open(stopwords_path, "r") as file:
            self.stop_words = set(file.read().splitlines())

        self.stemmer = Stemmer.default()

    def get_word_embeddings(self, text):
        text = self.preprocess_sentence(text)
        return [self.model.wv[word] for word in text if word in self.model.wv]

    def get_sentence_embeddings_from_word_embeddings(self, word_embeddings):
        return np.mean(word_embeddings, axis=0)

    def get_embeddings(self, text, preprocess=True):
        word_embeddings = self.get_word_embeddings(text)
        sentence_embedding = self.get_sentence_embeddings_from_word_embeddings(word_embeddings)
        return sentence_embedding


    def preprocess_sentence(self, sentence):

        sentence = str(sentence)
        # Convert to lowercase
        sentence = sentence.lower()
    
        # Remove special characters and numbers
        # (although numbers may be useful in some cases)
        # (for time constraints, I don't experiment on this)
        sentence = re.sub(r'\d+|\W+', ' ', sentence)
        
        # Tokenize the sentence
        words = word_tokenize(sentence)
        
        # Remove stopwords and apply stemming
        processed_words = [self.stemmer(word) for word in words if word not in self.stop_words]
        
        return processed_words
        