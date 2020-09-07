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


# Дополнение и усечение последовательности токенов
def pad_trunc(data, maxlen):
    """
    Дополнение для указанного набора данных
    нулевыми векторами или усечения до maxlen

    Так можно реализовать еще:
    [smp[:maxlen] + [[0.]*emb_dim] * (maxlen - len(smp)) for smp in data]
    """
    new_data = []
    # Создаём вектор нулей такой же длины,
    # что и у наших векторов слов
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            # Присоединяем к списку соответствующее
            # количество нулевых векторов
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data


# Оптимизация размера вектора идеи
def test_len(data, maxlen):
    total_len = truncated = exact = padded = 0
    for sample in data:
        total_len += len(sample)
        if len(sample) > maxlen:
            truncated += 1
        elif len(sample) < maxlen:
            padded += 1
        else:
            exact += 1
    print('Padded: {}'.format(padded))
    print('Equal: {}'.format(exact))
    print('Truncated: {}'.format(truncated))
    print('Avg length: {}'.format(total_len/len(data)))


if __name__ == '__main__':
    # Параметры CNN (c. 286)
    maxlen = 400
    batch_size = 32
    embedding_dims = 300
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 2

    dataset = pre_process_data('vk_comment_model')
    print(dataset[0])

    vectorized_data = vectorize(dataset)
    expected = collect_expected(dataset)

    # Разбиение на тренировочные/тестовые данные
    split_point = int(len(vectorized_data) * .8)
    x_train = vectorized_data[:split_point]
    y_train = expected[:split_point]
    x_test = vectorized_data[split_point:]
    y_test = expected[split_point:]

    # Выбор maxlen
    test_len(vectorized_data, 400)  # Надо брать среднюю длины

    # Собираем наши дополненные и усеченные данные
    x_train = pad_trunc(x_train, maxlen)
    x_test = pad_trunc(x_test, maxlen)

    x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
    y_train = np.array(y_train)
    x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
    y_test = np.array(y_test)