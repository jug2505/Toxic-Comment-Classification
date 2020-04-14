import pickle
from nltk.corpus.reader.api import CorpusReader


class PickledCommentsReader(CorpusReader):

    PKL_PATTERN = r'.*\.pickle'

    def __init__(self, root, fileids=PKL_PATTERN):
        CorpusReader.__init__(self, root, fileids)

    def texts_scores(self, fileids=None):
        """
        Возвращает документ, прочитанный из .pickle файла
        Использую генератор, чтобы сэкономить память
        """
        for path, encoding, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as file_read:
                yield pickle.load(file_read)

    def comments(self, fileids=None):
        """
        Возвращает текст комментариев (С тегами)
        """
        for text, score in self.texts_scores(fileids):
            yield text

    def scores(self, fileids=None):
        """
        Возвращает оценки
        """
        for text, score in self.texts_scores(fileids):
            yield score

    def paras(self, fileids=None):
        """
        Возвращает генератор абзацев
        """
        for comment in self.comments(fileids):
            for paragraph in comment:
                yield paragraph

    def sents(self, fileids=None):
        """
        Возвращает генератор предложений
        """
        for paragraph in self.paras(fileids):
            for sent in paragraph:
                yield sent

    def tagged(self, fileids=None):
        """
        Возвращает пару слово-тэг
        """
        for sentence in self.sents():
            for token in sentence:
                yield token

    def words(self, fileids=None):
        """
        Возвращает слова
        """
        for token in self.tagged(fileids):
            yield token[0]


if __name__ == '__main__':
    reader = PickledCommentsReader('corpus_proc')
    print(len(list(reader.comments())))
