# mult_percep.py
# Создаётся конвейер обучения Pipeline
# Происходит нормализация, векторизация и обучение

import numpy as np
# Обёртка для засекания времени
from functools import wraps
import time

# Библиотека Scikit-Learn
# Для сохранения модели
import joblib
# Разделять на тестовые и обучающие данные
from sklearn.model_selection import train_test_split
# Для создания матрицы ошибок
from sklearn.metrics import confusion_matrix
# Метрики
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Для визуализации
import itertools
import matplotlib.pyplot as plt

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
    Не токсичное : 0.0 < y <= 5.0
    Токсичное    : 5.0 < y <= 10.0
    """
    return np.digitize(continuous(corpus), [0.0, 5.0, 11.0])


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrap


def get_metrics(y_test, y_predicted):
    """
    Вычисляет метрики и возращает их
    """
    # Метрика точности
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')
    # Метрика полноты
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')
    # Метрика F1
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    # Метрика качества
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def plot_confusion_matrix(cm, classes, title="Матрица ошибок", cmap=plt.cm.winter):
    """
    Визуализация матрицы ошибок с помощью Matplotlib
    cm - матрица ошибок
    cmap - стиль оформления
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('Настоящая метка', fontsize=30)
    plt.xlabel('Предсказанный класс', fontsize=30)

    return plt


def corpus_plot(path):
    """
    Вывод графика количества документов в
    корпусе согласно их разметке
    """
    corpus = PickledCommentsReader(path)
    x = documents(corpus)
    y = continuous(corpus)
    data = np.arange(11)

    for i in y:
        for q in range(0, 11):
            if i == q:
                data[q] = data[q] + 1
    plt.bar(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], data, color='#539caf', align='center')
    plt.show()


def corpus_plot_categor(path):
    """
    Вывод графика количества документов в корпусе
    согласно метрике их классификации
    """
    corpus = PickledCommentsReader(path)
    x = documents(corpus)
    y = make_categorical(corpus)
    k = 0
    j = 0
    for i in y:
        if i <= 1:
            k = k + 1
        else:
            j = j + 1
    plt.bar(["Нетоксичные", "Токсичные"], [k, j], color='#539caf', align='center')
    plt.show()


@timeit
def train_model(path, model, contin=False, saveto=None, cv=12):
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

    # Разделён ли уже на категории
    if contin:
        y = continuous(corpus)
    else:
        y = make_categorical(corpus)

    # Разделение на обучающую выборку и выборку для проверки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # Обучение модели
    model.fit(X_train, y_train)

    # Предсказание классов
    y_predicted = model.predict(X_test)
    # Вычисление оценок
    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted)
    # Вывод оценок в консоль
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    # Вычисление матрицы ошибок
    cm = confusion_matrix(y_test, y_predicted)
    # Визуализация матрицы ошибок
    figure = plt.figure(figsize=(10, 10))
    plot = plot_confusion_matrix(cm, classes=['Нетоксичное', 'Токсичное'])
    plt.show()
    # Запись на диск
    if saveto:
        joblib.dump(model, saveto)


if __name__ == '__main__':
    # Pipeline для создания конвейера обучения
    from sklearn.pipeline import Pipeline
    # Multi-layer Perceptron classifier
    # Будем решать задачу классификации
    from sklearn.neural_network import MLPClassifier
    # TF-IDF оценка важности слова (term frequency - inverse document frequency)
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    # Нормализатор из файла transformer
    from transformer import TextNormalizer

    # Путь к обработанному pickle-корпусу
    corpath = "corpus_proc"
    # По этому пути будет сохранена модель
    modelpath = "model.pickle"

    corpus_plot(corpath)
    corpus_plot_categor(corpath)

    # Использую конвейер pipeline для удобства
    # связывания нормализаторов и классификатора

    pipeline = Pipeline([
        ('Normalize', TextNormalizer()),
        ('Vectorize', TfidfVectorizer()),
        ('Classify', MLPClassifier(hidden_layer_sizes=[500, 150], verbose=True))
    ])

    print('Начало обучения')
    scores, delta = train_model(corpath, pipeline, contin=False, saveto=modelpath)
    print('Конец обучения')

    # Вывод точностей модели
    print('Время обучения: {} сек'.format(delta))
    print('Путь к модели: {}'.format(modelpath))
