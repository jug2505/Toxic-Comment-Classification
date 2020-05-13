# preprocessor.py
# Реализация класса первичной обработки корпуса
# Сохранияет корпус в pickle формате

import os
import nltk
import pickle


class Preprocessor:
    """
    Класс является обёрткой над JsonCorpusReader
    Проводит токенизацию и сохраняет в формат pickle,
    который потом может считать PickledCorpusReader.
    (Для компактности и удобства)
    """

    def __init__(self, corpus, target=None):
        """
        corpus - 'JsonCorpusReader'
        target - папка куда экспортируем
        """
        self.target = target
        self.corpus = corpus

    def target(self, path):
        # Нужно снормализовать путь и сделать его абслолютным
        # На всякий случай заменим ещё переменные окружения и пользователя
        path = os.path.expandvars(path)
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if os.path.exists(path):
            if not os.path.isdir(path):
                print("Directory is needed - Preprocessor")
            else:
                self.target = path

    def abspath(self, fileid):
        """
        Возвращает абсолютный путь для файла на запись
        """
        # Найдём путь относительно корня корпуса
        parent = os.path.relpath(os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root)
        # Вычисляем имя
        basename = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)
        # Меняем расширение на pickle
        basename = name + ".pickle"
        # Возвращаем путь относительно target
        return os.path.normpath(os.path.join(self.target, parent, basename))

    def tokenize(self, fileid):
        """
        Производит:
        - Чтение
        - Сегментацию на предложения sent_tokenize
        - Токенизацию предложений word_tokenize
        - Проставление тегов pos_tag
        Возвращает список предложений, который является списком
        слов с тегами
        """
        # Возвращает генератор, поэтому такой странный синтаксис
        comment = list(self.corpus.comments(fileids=fileid))[0]
        score = comment["overall"]
        comment_text = comment["commentText"]
        # Тегизация
        comment_tagged = []
        for sentence in nltk.sent_tokenize(comment_text, language='russian'):
            comment_tagged.append(nltk.pos_tag(nltk.word_tokenize(sentence, language='russian'), lang='rus'))

        return [comment_tagged, score]

    def process(self, fileid):
        """
        Для одного файла производит:
        - Проверку на существование
        - Вызов tokenize
        - Запись в .pickle
        """
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        # Проверка существования
        if not os.path.exists(parent):
            os.makedirs(parent)

        # Вызов tokenize
        doc = self.tokenize(fileid)
        # Выгружаем
        with open(target, 'wb') as file_write:
            pickle.dump(doc, file_write, pickle.HIGHEST_PROTOCOL)

        del doc
        return target

    def transform(self):
        """
        Собственно обработка
        """
        # Создадим директорию, если её ещё нет
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        for fileid in self.corpus.fileids():
            yield self.process(fileid)


if __name__ == '__main__':
    from comments_reader import JsonCorpusReader

    corpus = JsonCorpusReader('corpus_marked')
    transformer = Preprocessor(corpus, 'corpus_proc')
    documents = transformer.transform()
    print(len(list(documents)))
