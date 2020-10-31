# Toxic-Comment-Classification

corpus/vk_comment_parser.py - выдаёт комментарии под постом В Контакте в виде json файлов.

Создать директорию corpus_marked и хранить там размеченный корпус json файлов.

Вызов скрипта word2vec_model.py создаёт Word2Vec модель обученную на корпусе.

После этого можно запускать скрипты создания классификаторов: 
* convoltuional.py - На основе свёрточной нейросети
* recurent.py - На основе рекурррентной нейросети
* double_recurent.py - На основе двунаправленной нейросети
* lstm.py - На основе нейросети с  долгой краткосрочной памятью

Ссылка на размеченный корпус: https://www.dropbox.com/s/8qomv4ne10zl7x6/corpus_marked.zip?dl=0
