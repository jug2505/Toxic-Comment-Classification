# lstm.py
# Реализация нейросети с долгой краткосрочной памятью
#  для анализа "языка вражды" в сообщениях
# Реализация на Keras для анализа тональности

# Импорт
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.models import model_from_json
import nltk.tokenize

# Препроцессор
from comments_reader import JsonCorpusReader
from transformer import TextNormalizer
from random import shuffle
from preprocessor import Preprocessor

# Для векторизатора
from gensim.models.word2vec import Word2Vec
# Для создания матрицы ошибок
from sklearn.metrics import confusion_matrix
# Метрики
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Для визуализации
import itertools
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    maxlen = 200
    batch_size = 32
    embedding_dims = 300
    epochs = 10
    num_neurons = 50

    prep = Preprocessor('corpus_marked', 'vk_comment_model')
    x_train, y_train, x_test, y_test = prep.train_pipeline(maxlen, embedding_dims)

    # LSTM
    model = Sequential()

    model.add(LSTM(
        num_neurons, return_sequences=True,
        input_shape=(maxlen, embedding_dims)
    ))

    model.add(Dropout(.2))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Обучение и сохранение модели
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test
                         ))

    # Предсказание классов
    y_predicted = np.around(model.predict(x_test))
    # Вычисление оценок
    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted)
    # Вывод оценок в консоль
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    # Вычисление матрицы ошибок
    cm = confusion_matrix(y_test, y_predicted)
    # Визуализация матрицы ошибок
    figure = plt.figure(figsize=(10, 10))
    plot = plot_confusion_matrix(cm, classes=['Неоскорбительное', 'Оскорбительное'])
    plt.show()

    model_structure = model.to_json()  # Сохранение структуры
    with open("lstm_model.json", "w") as json_file:
        json_file.write(model_structure)
    model.save_weights("lstm_weights.h5")  # Сохранение обученной модели (весов)

    with open("lstm_model.json", "r") as json_file:
        json_string = json_file.read()
    model = model_from_json(json_string)
    model.load_weights('lstm_weights.h5')

    # Тестовый пример
    sample_1 = "темная тема"

    # Предсказание
    test_vec = Preprocessor('corpus_marked', 'vk_comment_model') \
        .exm_pipeline(sample_1, maxlen, embedding_dims)

    print()
    print(model.predict(test_vec))
    print()
    print(np.around(model.predict(test_vec)))
