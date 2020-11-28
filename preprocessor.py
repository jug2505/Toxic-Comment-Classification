# preprocessor.py
# Предварительная обработка
# Импорт
import numpy as np
import nltk.tokenize
# Препроцессор
from comments_reader import JsonCorpusReader
from transformer import TextNormalizer
from random import shuffle
# Для векторизатора
from gensim.models.word2vec import Word2Vec


class Preprocessor:

    def __init__(self, corpus_name, model_name):
        self.corpus_name = corpus_name  # corpus_marked
        self.model_name = model_name  # 'vk_comment_model'

    @staticmethod
    def documents(corpus):
        """
        Извлекает документы из объекта чтения корпуса
        """
        return list(TextNormalizer().fit_transform(corpus.words()))

    @staticmethod
    def continuous(corpus):
        """
        Функция для получения исходного числового рейтинга
        """
        return list(corpus.scores())

    def make_categorical(self, corpus):
        """
        Не токсичное : 0.0 < y <= 5.0
        Токсичное    : 5.0 < y <= 10.0
        """
        return np.digitize(self.continuous(corpus), [5.0, 11.0])

    def pre_process_data(self, contin=False):
        """
        Загрузка данных и разделение на категории
        """
        # Загружаем данные из корпуса
        corpus = JsonCorpusReader(self.corpus_name)
        x = self.documents(corpus)

        # Разделён ли уже на категории
        if contin:
            y = self.continuous(corpus)
        else:
            y = self.make_categorical(corpus)

        dataset_prep = []
        for i in range(len(x)):
            dataset_prep.append((y[i], x[i]))

        shuffle(dataset_prep)
        return dataset_prep

    def vectorize(self, dataset):
        """
        Векторизация
        """
        model_vec = Word2Vec.load(self.model_name)
        word_vectors = model_vec.wv

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

    @staticmethod
    def collect_expected(dataset):
        """
        Выбираем целевые значения из набора данных
        """
        expected_set = []
        for sample in dataset:
            expected_set.append(sample[0])
        return expected_set

    @staticmethod
    def pad_trunc(data, maxlen):
        """
        Дополнение и усечение последовательности токенов
        Дополнение для указанного набора данных
        нулевыми векторами или усечения до maxlen
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

    @staticmethod
    def test_len(data, maxlen):
        """
        Оптимизация размера вектора идеи
        """
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
        print('Avg length: {}'.format(total_len / len(data)))

    @staticmethod
    def prep_exm(sent):
        """
        Препроцессор для тествых примеров
        """
        sent_arr = []
        for word in nltk.word_tokenize(sent, language='russian'):
            sent_arr.append(word)
        return list(TextNormalizer().normalize(sent_arr))

    def train_pipeline(self, maxlen, embedding_dims):  # maxlen 100 , embedding_dims 300
        dataset = self.pre_process_data()

        vectorized_data = self.vectorize(dataset)
        expected = self.collect_expected(dataset)

        # Разбиение на тренировочные/тестовые данные
        split_point = int(len(vectorized_data) * .8)
        x_train = vectorized_data[:split_point]
        y_train = expected[:split_point]
        x_test = vectorized_data[split_point:]
        y_test = expected[split_point:]

        # Собираем наши дополненные и усеченные данные
        x_train = self.pad_trunc(x_train, maxlen)
        x_test = self.pad_trunc(x_test, maxlen)

        x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
        y_train = np.array(y_train)
        x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
        y_test = np.array(y_test)

        return x_train, y_train, x_test, y_test

    def exm_pipeline(self, sample, maxlen, embedding_dims):
        sample_data = self.prep_exm(sample)
        vec_list = self.vectorize([(0, sample_data)])
        test_vec_list = self.pad_trunc(vec_list, maxlen)
        return np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))


if __name__ == '__main__':
    # Выбор maxlen
    prep = Preprocessor('corpus_marked', 'vk_comment_model')
    dataset = prep.pre_process_data()
    vectorized_data = prep.vectorize(dataset)
    Preprocessor.test_len(vectorized_data, 100)  # Надо брать среднюю длины
