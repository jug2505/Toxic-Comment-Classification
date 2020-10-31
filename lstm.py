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

if __name__ == '__main__':
    maxlen = 200
    batch_size = 32
    embedding_dims = 300
    epochs = 5
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
    print(np.argmax(model.predict(test_vec), axis=-1))
