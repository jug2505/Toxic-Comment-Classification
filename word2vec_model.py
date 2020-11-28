# word2vec_model.py
# Создание предметной модели word2vec
from comments_reader import JsonCorpusReader
from transformer import TextNormalizer
# Для обучения
from gensim.models.word2vec import Word2Vec
# Для тестов
from nltk.stem.snowball import SnowballStemmer


if __name__ == '__main__':
    # Предварительная обработка
    corpus = JsonCorpusReader('corpus_marked')
    # Формат [["слова"], ["слова"]]
    token_list = list(TextNormalizer().fit_transform(corpus.words()))

    # Параметры
    num_features = 300
    min_word_count = 1
    num_workers = 4
    window_size = 6
    subsampling = 1e-3

    # Создание экземпляра
    model = Word2Vec(
        token_list,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=window_size,
        sample=subsampling
    )

    # Заморозка модели, исключение ненужных выходных весов
    model.init_sims(replace=True)

    # Сохранение
    model_name = "vk_comment_model"
    model.save(model_name)

    # Загрузка
    model_name = "vk_comment_model"
    model = Word2Vec.load(model_name)

    stemmer = SnowballStemmer('russian')
    print(model.wv.similarity(stemmer.stem("поезд"), stemmer.stem("Пусан")))
    print(model.wv[stemmer.stem("поезд")])
