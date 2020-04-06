
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



