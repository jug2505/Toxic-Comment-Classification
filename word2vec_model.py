# Создание собственной модели word2vec
from comments_reader import JsonCorpusReader

# TODO:Добавить предварительную обработку
if __name__ == '__main__':
    # Предварительная обработка (с. 233)
    corpus = JsonCorpusReader('corpus_marked')
    # Формат [["слова"], ["слова"]]
    token_list = list(corpus.texts())

    # Обучение
    from gensim.models.word2vec import Word2Vec

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
    from gensim.models.word2vec import Word2Vec

    model_name = "vk_comment_model"
    model = Word2Vec.load(model_name)
    print(model.wv.similarity("привет", "андрей"))
