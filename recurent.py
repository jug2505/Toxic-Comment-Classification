# recurent.py
# Реализация рекуррентной нейросети для анализа
# "языка вражды" в сообщениях
# Реализация на Keras для анализа тональности

# Импорт
import numpy as np
from keras.layers import SimpleRNN
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.models import model_from_json
# Препроцессор
from preprocessor import Preprocessor


if __name__ == '__main__':
    # Параметры RNN
    maxlen = 100
    batch_size = 32
    embedding_dims = 300
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 5
    num_neurons = 25

    prep = Preprocessor('corpus_marked', 'vk_comment_model')
    x_train, y_train, x_test, y_test = prep.train_pipeline(maxlen, embedding_dims)

    # Инициплизация пустой сети Keras
    model = Sequential()

    # Добавление рекррентного слоя
    model.add(SimpleRNN(
        num_neurons, return_sequences=True,
        input_shape=(maxlen, embedding_dims)
    ))

    # Добавление слоя дропаута
    model.add(Dropout(.2))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Компиляция нашей рекуррентной нейронной сети
    model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Обучение и сохранение модели
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test
                         ))

    # Сохранение результатов
    model_structure = model.to_json()  # Сохранение структуры
    with open("rec_model.json", "w") as json_file:
        json_file.write(model_structure)
    model.save_weights("rec_weights.h5")  # Сохранение обученной модели (весов)

    # Применение модели в конвейере
    # Загрузка сохраненной модели
    with open("rec_model.json", "r") as json_file:
        json_string = json_file.read()
    model = model_from_json(json_string)
    model.load_weights('rec_weights.h5')

    # Тестовый пример
    sample_1 = "темная тема"

    # Предсказание
    test_vec = Preprocessor('corpus_marked', 'vk_comment_model') \
        .exm_pipeline(sample_1, maxlen, embedding_dims)

    print()
    print(model.predict(test_vec))
    print()
    print(np.argmax(model.predict(test_vec), axis=-1))
