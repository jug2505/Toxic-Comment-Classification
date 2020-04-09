
import numpy as np
# Обёртка для засекания времени
from functools import wraps
import time

# Библиотека Scikit-Learn
# Для сохранения модели
from sklearn.externals import joblib
# Для перекрёстной проверки
from sklearn.model_selection import cross_val_score


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
    TODO: Добавить метрики
    """
    return np.digitize(continuous(corpus), ["TODO: Список из метрики"])


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrap


@timeit
def train_model(path, model, continuous=True, saveto=None, cv=12):
    """
    Обучает модель на корпусе по пути path
    Вычисляет оценки перекрёстной проверки, используя параметр cv
    Обучает модель на всём объёме данных. Возвращает оценки.
    Запись модели по пути saveto.
    continuous = True - нужно вызвать функцию continuous,
    чтобы получить входные значения X и целевые значения Y
    """
    # Загружаем данные из корпуса TODO: класс PickledCommentsReader
    corpus = PickledCommentsReader(path)
    X = documents(corpus)
    # Разделён ли уже на категории
    if continuous:
        Y = continuous(corpus)
        score = 'r2_score'
    else:
        Y = make_categorical(corpus)
        score = 'f1_score'

    # Вычислим оценки
    scores = cross_val_score(model, X, Y, cv=cv)
    # Обучаем модель
    model.fit(X, Y)

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

    # TODO: файлы см. архитектуру проекта
    from reader import PickledReviewReader
    from transformer import TextNormalizer

    # TODO: пути
    corpath = '../corpuspathname'
    modelpath = 'a.pkl'

    # Использую конвейер pipeline для удобства
    # связывания трансформаторов и классификатора
    pipeline = Pipeline([
        ('Normalize', TextNormalizer()),
        ('Vectorize', TfidfVectorizer()),
        ('Classify', MLPClassifier(hidden_layer_sizes=[500, 150], verbose=True))
    ])

    print('Start training')
    scores, delta = train_model(corpath, pipeline, continuous=False, saveto=modelpath)
    print('Train complete')

    # Вывод точностей модели
    for index, score in enumerate(scores):
        print('Accuracy on part №{}: {}'.format((index + 1), score))
    print('Learning time: {} sec'.format(delta))
    print('Model path: {}'.format(modelpath))

