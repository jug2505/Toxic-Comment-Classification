# Реализация на Keras для анализа тональности
# Импорт (c. 283)
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D

# Препроцессор
from comments_reader import JsonCorpusReader
from transformer import TextNormalizer
from random import shuffle

# Для векторизатора
from gensim.models.word2vec import Word2Vec


def documents(corpus):
    """
    Извлекает документы, маркированные частями речи,
    из объекта чтения корпуса
    """
    return list(TextNormalizer().fit_transform(corpus.words()))


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


def pre_process_data(corpus_name, contin=False):  # corpus_marked
    # Загружаем данные из корпуса
    corpus = JsonCorpusReader(corpus_name)
    x = documents(corpus)

    # Разделён ли уже на категории
    if contin:
        y = continuous(corpus)
    else:
        y = make_categorical(corpus)

    dataset = []
    for i in range(len(x)):
        dataset.append((y[i], x[i]))

    shuffle(dataset)
    return dataset


def vectorize(dataset, model_name='vk_comment_model'):
    model_name = "vk_comment_model"
    model = Word2Vec.load(model_name)
    word_vectors = model.wv

    vectorized_data = []
    for sample in dataset:
        tokens = sample[1]
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass  # В словаре w2v нет соответствующего токена
        vectorized_data.append(sample_vecs)
    return vectorized_data


def collect_expected(dataset):
    """ Выбираем целевые значения из набора данных """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected
