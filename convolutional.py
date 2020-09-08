# Реализация на Keras для анализа тональности
# Импорт (c. 283)
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D

import nltk.tokenize

# Препроцессор
from comments_reader import JsonCorpusReader
from transformer import TextNormalizer
from random import shuffle

# Для векторизатора
from gensim.models.word2vec import Word2Vec


def documents(corpus):
    """
    Извлекает документы из объекта чтения корпуса
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

    dataset_prep = []
    for i in range(len(x)):
        dataset_prep.append((y[i], x[i]))

    shuffle(dataset_prep)
    return dataset_prep


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


# Препроцессор для тествых примеров
def prep_exm(sent):
    sent_arr = []
    for word in nltk.word_tokenize(sent, language='russian'):
        sent_arr.append(word)
    return list(TextNormalizer().normalize(sent_arr))


if __name__ == '__main__':
    # Параметры CNN (c. 286)
    maxlen = 400
    batch_size = 32
    embedding_dims = 300
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 2

    dataset = pre_process_data('corpus_marked')
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

    ######################################################
    ### Архитектура сверточной нейронной сети (c. 288) ###

    # Задание нач. значения генератора случайных чисел,
    # если нужно выбирать одинаковые начальные веса для нейронов
    # Для отладки
    import numpy as np

    np.random.seed(1337)

    # Формируем одномерную CNN
    print('Building model ...')
    model = Sequential()
    model.add(Conv1D(
        filters,
        kernel_size,
        padding='valid',
        activation='relu',
        strides=1,
        input_shape=(maxlen, embedding_dims)
    ))

    # Субдискретизация
    model.add(GlobalMaxPooling1D())

    # Полносвязный слой с ДРОПАУТОМ (c. 291) !!!
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # Процеживание
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Компиляция CNN (c. 292)
    model.compile(
        loss='binary_crossentropy',  # Почитать про binary_crossentropy и categorical_crossentropy
        optimizer='adam',
        metrics=['accuracy']
    )

    # Выходной слой для дискретной переменной
    # При categorical_crossentropy (когда несколько классов)
    # model.add(Dense(num_classes))
    # model.add(Activation('sigmoid'))

    # Обучение CNN
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test)
    )

    # Сохранение результатов (с. 294)
    model_structure = model.to_json()  # Сохранение структуры
    with open("cnn_model.json", "w") as json_file:
        json_file.write(model_structure)
    model.save_weights("cnn_weights.h5")  # Сохранение обученной модели (весов)

    ### Применение модели в конвейере (c. 296)
    # Загрузка сохраненной модели
    from keras.models import model_from_json

    with open("cnn_model.json", "r") as json_file:
        json_string = json_file.read()
    model = model_from_json(json_string)
    model.load_weights('cnn_weights.h5')

    # Тестовый пример
    sample_1 = "Как же я обожаю уезжать в 20:10 из института"

    # Предсказание
    sample_data = prep_exm(sample_1)
    vec_list = vectorize([(1, sample_data)])
    test_vec_list = pad_trunc(vec_list, maxlen)
    test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
    print(model.predict(test_vec))
    print(model.predict_classes(test_vec))
