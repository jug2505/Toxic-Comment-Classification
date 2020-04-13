import os


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

    # TODO:  def fileids(self, fileids=None):

    def abspath(self, fileid):
        """
        Возвращает абсолютный путь для файла на запись
        """
        # Найдём
        parent = os.path.relpath(os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root)

