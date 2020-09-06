# comments_reader.py
# Реализация класса JsonCorpusReader
# Считывает корпус в память с диска

# Кодировки, чтобы не было проблем с языками
import codecs
# Комментарии храню в json
import json
import nltk
# Будем наследоваться
from nltk.corpus.reader.api import CorpusReader


class JsonCorpusReader(CorpusReader):

    DOC_PATTERN = r'.*\.json'

    def __init__(self, root, fileids=DOC_PATTERN):
        """
        Инициальзируем класс чтения
        """
        CorpusReader.__init__(self, root, fileids)

    def comments(self, fileids=None):
        """
        Возвращает генератор (для экономии памяти).
        Полный текст одного json файла
        (т.е. одного комментария)
        """
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as read_file:
                yield json.load(read_file)

    def texts(self):
        """
        Возвращает полный текст комментария (Генератор)
        """
        for comment in self.comments():
            yield comment["commentText"]

    def scores(self):
        """
        Возвращает оценку (Генератор)
        """
        for comment in self.comments():
            yield comment["overall"]

    def ids(self):
        """
        Возвращает id комментария (Генератор)
        """
        for comment in self.comments():
            yield comment["id"]

    def ids_scores_texts(self):
        """
        Возвращает кортеж всего (Генератор)
        """
        for comment in self.comments():
            yield comment["id"], comment["overall"], comment["commentText"]

    def sents(self):
        """
        Возвращает генератор предложений
        """
        for text in self.texts():
            # Выделяем предложения с пом-ю nltk
            for sentence in nltk.sent_tokenize(text, language='russian'):
                yield sentence

    def words(self):
        """
        Возвращает список предложений состоящий из
        списков слов (Генератор)
        """
        for sentence in self.sents():
            sent_arr = []
            for word in nltk.word_tokenize(sentence, language='russian'):
                sent_arr.append(word)
            yield sent_arr

    def tagged_sents(self):
        """
        Возвращает предложения с тегами частей речи.
        Используется сет Russian National Corpus
        """
        for sentence in self.sents():
            yield nltk.pos_tag(nltk.word_tokenize(sentence, language='russian'), lang='rus')


if __name__ == '__main__':
    corpus = JsonCorpusReader('corpus_marked')
    print(list(corpus.words()))
