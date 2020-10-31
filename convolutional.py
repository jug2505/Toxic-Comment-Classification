# convolitional.py
# Реализация сверточной нейросети для анализа
# "языка вражды" в сообщениях
# Импорт
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
import numpy as np
from keras.models import model_from_json

# Препроцессор
from preprocessor import Preprocessor


if __name__ == '__main__':
    # Задание нач. значения генератора случайных чисел,
    # если нужно выбирать одинаковые начальные веса для нейронов
    # для отладки
    np.random.seed(1337)

    # Параметры CNN
    maxlen = 100
    batch_size = 32
    embedding_dims = 300
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 10

    prep = Preprocessor('corpus_marked', 'vk_comment_model')
    x_train, y_train, x_test, y_test = prep.train_pipeline(maxlen, embedding_dims)

    # Архитектура сверточной нейронной сети
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

    # Полносвязный слой с дропаутом
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # Процеживание
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Компиляция CNN
    model.compile(
        loss='binary_crossentropy',  # TODO: Почитать про binary_crossentropy и categorical_crossentropy
        optimizer='adam',
        metrics=['accuracy']
    )

    # Обучение CNN
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test)
    )

    # Сохранение результатов
    model_structure = model.to_json()  # Сохранение структуры
    with open("cnn_model.json", "w") as json_file:
        json_file.write(model_structure)
    model.save_weights("cnn_weights.h5")  # Сохранение обученной модели (весов)

    # Применение модели в конвейере
    # Загрузка сохраненной модели

    with open("cnn_model.json", "r") as json_file:
        json_string = json_file.read()
    model = model_from_json(json_string)
    model.load_weights('cnn_weights.h5')

    # Тестовый пример
    sample_1 = "Темная тема"

    # Предсказание
    test_vec = Preprocessor('corpus_marked', 'vk_comment_model')\
        .exm_pipeline(sample_1, maxlen, embedding_dims)

    print()
    print(model.predict(test_vec))
    print()
    print(np.argmax(model.predict(test_vec), axis=-1))
