import nltk
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer


# Класс для нормализации текста
class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='russian'):
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.stemmer = SnowballStemmer(language)

    def is_punct(self, token):
        """
        Сравнивает первую букву в названии категориии Юникода каждого
        символа c P (Punctuation)
        """
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def is_nonlex(self, tag):
        return tag == 'NONLEX'

    def stemming(self, token):
        return self.stemmer.stem(token)

    def normalize(self, document):
        """
        Нормализация стеммингом
        """
        return [
            self.stemming(token)
            for sent in document
            for token, tag in sent
            if not self.is_nonlex(tag) and not self.is_stopword(token) and not self.is_punct(token)
        ]

    # fit и transform для pipeline
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        return [' '.join(self.normalize(document)) for document in documents]


if __name__ == '__main__':
    from corpus_reader import PickledCommentsReader

    corpus = PickledCommentsReader("corpus_proc")
    docs = list(TextNormalizer().fit_transform(corpus.comments()))
    for doc in docs:
        print(doc)
