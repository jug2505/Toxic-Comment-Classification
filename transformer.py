# transformer.py
# Класс нормализации текста
# Используется стеммер Портера (Snowball)

import nltk
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer


# Класс для нормализации текста
class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('russian'))
        self.stemmer = SnowballStemmer('russian')

    @staticmethod
    def is_punct(token):
        """
        Сравнивает первую букву в названии категориии Юникода каждого
        символа c P (Punctuation)
        """
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token):
        """
        Является ли токен стоп-словом
        """
        return token.lower() in self.stopwords

    def stemming(self, token):
        """
        Непосредственно стемминг
        """
        return self.stemmer.stem(token)

    def normalize(self, sent):
        """
        Нормализация стеммингом
        """
        return [
            self.stemming(token)
            for token in sent
            if not self.is_stopword(token) and not self.is_punct(token)
        ]

    # fit и transform для pipeline
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        norm_corp = []
        for comment in documents:
            norm_corp.append(self.normalize(comment))
        return norm_corp


if __name__ == '__main__':
    from comments_reader import JsonCorpusReader

    corpus = JsonCorpusReader("corpus_marked")
    docs = list(TextNormalizer().fit_transform(corpus.words()))
    print(docs)
