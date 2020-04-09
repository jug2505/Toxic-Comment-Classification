import vk  # Импортируем модуль vk
import os


# Функция сохранения базы в json
def save_data(comments, post_id):
    dir = str(post_id)
    os.mkdir(dir)
    for comment in comments:
        with open(str(comment)+'.json', 'w') as write_file:




if __name__ == "__main__":
    # Сервисный ключ доступа
    token = input("Enter token: ")
    session = vk.Session(access_token=token)  # Авторизация
    vk_api = vk.API(session)

    # Команда ВК
    owner_id = 22822305
    post_id = 1061966
    comments = vk.API.wall.getComments(owner_id=owner_id, post_id=post_id, need_likes=0, count=100)

