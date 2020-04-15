import numpy as np
# Обёртка для засекания времени
from functools import wraps
import time

# Библиотека Scikit-Learn
# Для сохранения модели
import joblib
# Для перекрёстной проверки
from sklearn.model_selection import cross_val_score

from corpus_reader import PickledCommentsReader


def documents(corpus):
    """
    Извлекает документы, маркированные частями речи,
    из объекта чтения корпуса
    """
    return list(corpus.comments())


def continuous(corpus):
    """
    Функция для получения исходного числового рейтинга
    """
    return list(corpus.scores())


def make_categorical(corpus):
    """
    not toxic : 0.0 < y <= 5.0
    toxic     : 5.0 < y <= 10.0
    """
    return np.digitize(continuous(corpus), [0.0, 5.0, 10.0])


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrap


@timeit
def train_model(path, model, contin=True, saveto=None, cv=12):
    """
    Обучает модель на корпусе по пути path
    Вычисляет оценки перекрёстной проверки, используя параметр cv
    Обучает модель на всём объёме данных. Возвращает оценки.
    Запись модели по пути saveto.
    contin = True - нужно вызвать функцию continuous,
    чтобы получить входные значения X и целевые значения Y
    """
    # Загружаем данные из корпуса
    corpus = PickledCommentsReader(path)
    X = documents(corpus)
    print(X)
    # Разделён ли уже на категории
    if contin:
        y = continuous(corpus)
        score = 'r2_score'
    else:
        y = make_categorical(corpus)
        score = 'f1_score'

    # Вычислим оценки TODO: scoring
    scores = cross_val_score(model, X, y, cv=cv)

    # Обучаем модель
    model.fit(X, y)

    # Запись на диск
    if saveto:
        joblib.dump(model, saveto)
    # Возврат оценки
    return scores


if __name__ == '__main__':
    from sklearn.pipeline import Pipeline
    # Multi-layer Perceptron classifier
    # Будем решать задачу классификации
    from sklearn.neural_network import MLPClassifier
    # TF-IDF оценка важности слова (term frequency - inverse document frequency)
    from sklearn.feature_extraction.text import TfidfVectorizer

    from transformer import TextNormalizer

    corpath = "corpus_proc"
    modelpath = "model.pickle"

    # Использую конвейер pipeline для удобства
    # связывания трансформаторов и классификатора

    pipeline = Pipeline([
        ('Normalize', TextNormalizer()),
        ('Vectorize', TfidfVectorizer()),
        ('Classify', MLPClassifier(hidden_layer_sizes=[500, 150], verbose=True))
    ])

    print('Start training')
    scores, delta = train_model(corpath, pipeline, contin=False, saveto=modelpath)
    print('Train complete')

    # Вывод точностей модели
    for index, score in enumerate(scores):
        print('Accuracy on part №{}: {}'.format((index + 1), score))
    print('Learning time: {} sec'.format(delta))
    print('Model path: {}'.format(modelpath))
