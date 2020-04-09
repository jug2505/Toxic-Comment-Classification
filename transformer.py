
import nltk
import unicodedata
from nltk.corpus import wordnet as wn
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.wordnet import WordNetLemmatizer


# Класс для нормализации текста
class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='russian'):
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    # Сравнивает первую букву в названии категориии Юникода каждого
    # символа c P (Punctuation)
    def is_punct(self, token):
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token):
        return token.lower() in self.stopwords

