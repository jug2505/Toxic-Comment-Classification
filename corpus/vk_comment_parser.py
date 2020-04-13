import vk_api  # Импортируем модуль vk
import os
import json


# Функция сохранения базы в json
def save_data(comments, post_id, count):
    os.mkdir(str(post_id))
    i = 0
    while i < count:
        data = {'overal': -1, 'commentText': comments['items'][i]['text']}
        with open(str(post_id) + '/' + str(comments['items'][i]['id']) + '.json', 'w') as write_file:
            json.dump(data, write_file, ensure_ascii=False)
        i = i + 1


if __name__ == "__main__":
    print('Login: ')
    login = input()
    print('Password: ')
    password = input()

    vk_session = vk_api.VkApi(login, password)
    vk_session.auth()
    vk = vk_session.get_api()

    # Команда ВК
    owner_id = str(-22822305)
    post_id = 1061966
    count = 50
    comments = vk.wall.getComments(owner_id=owner_id, post_id=post_id, need_likes=0, count=count)
    save_data(comments, post_id, count)