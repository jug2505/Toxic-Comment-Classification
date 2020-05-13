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
    print("Login: ")
    login = input()
    print("Password: ")
    password = input()

    vk_session = vk_api.VkApi(login, password)
    vk_session.auth()
    vk = vk_session.get_api()

    owner_id = str(-23064236)
    post_id = 2045993
    count = 100
    comments = vk.wall.getComments(owner_id=owner_id, post_id=post_id, sort="desc", count=count)
    save_data(comments, post_id, count)
