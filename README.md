# Toxic-Comment-Classification

corpus/vk_comment_parser.py - выдаёт комментарии под постом В Контакте в виде json файлов.

Создать директорию corpus_marked и хранить там размеченный корпус json файлов.

Вызов скрипта preprocessor.py преобразует данные в .pickle файлы, чтобы с ними было в дальнейшем легко работать. Хранятся в corpus_proc/

После этого запускается mult_percep.py, в котором происходит нормализация, векторизация и обучение.

Ссылка на размеченный корпус: https://www.dropbox.com/s/8qomv4ne10zl7x6/corpus_marked.zip?dl=0
