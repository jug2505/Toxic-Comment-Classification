# vk_comment_parser.py
# Парсер коммемнтариев под постом В Контакте
# Сохраняет данные в JSON формате

import vk_api  # Импортируем модуль vk
import os
import json


# Функция сохранения данных в json
def save_data(comments, post_id, count):
    os.mkdir(str(post_id))
    i = 0
    while i < count:
        data = {"overall": -1, "id": comments["items"][i]["id"], "commentText": comments["items"][i]["text"]}
        with open(str(post_id) + '/' + str(comments["items"][i]["id"]) + ".json", 'w') as write_file:
            json.dump(data, write_file, ensure_ascii=False)
        i = i + 1


if __name__ == "__main__":
    # Авторизация пользователя
    print("Enter your login (email, phone number): ")
    login = input()
    print("Password: ")
    password = input()

    # Создание активной сессии
    vk_session = vk_api.VkApi(login, password)
    vk_session.auth()
    vk = vk_session.get_api()

    # Максимальное число комментариев, получаемое с одного запроса
    count = 100
    # В пределах паблика спрашиваем каждый раз id поста
    program_run = True
    while program_run:
        # Цикл прохода по сообществам
        print("Enter id of the group:")
        owner_id = '-' + input()
        group_run = True
        # Цикл прохода по комментариям
        while group_run:
            print("Enter id of the post:")
            post_id = int(input())
            comments = vk.wall.getComments(
                owner_id=owner_id,
                post_id=post_id,
                sort="desc",
                count=count)
            save_data(comments, post_id, count)
            print("Press 'q' if you want to change group or another button if not")
            if input() == 'q':
                group_run = False

        print("Press 'q' if you want to leave the program \
            or another button to change the active group")
        if input() == 'q':
            program_run = False
