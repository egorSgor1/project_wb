# Wildberries project 

Данный репозиторий представляет собой решение 5-го этапа дипломного проекта Техношколы
Data Scientist WB по направлению 'Репутация пользователей'


## Overview
 
В рамках данного этапа я реализовал микросервис в виде web-приложения, который принимает
пользовательские запросы с одним или несколькими изображениями и классифицирует их
как допустимый материал / спам на основе модели, полученной мною на 4-ом этапе
(она загружена на Hugging Face и доступна по [ссылке](https://huggingface.co/EGORsGOR/vit-spam)).
Docker-образ моего решения загружен на Docker Hub
([ссылка](https://hub.docker.com/r/egornov/project_wb)).

В папке /src находятся исходные коды решения, все необходимые зависимости,
а также Dockerfile, позволяющий получить из всего этого docker-образ. 
При этом для запуска контейнера с приложением образ не собирается заново,
а берется с Docker Hub.  


## Инструкции

### Сборка репозитория 
```commandline
git clone https://github.com/egorSgor1/project_wb.git
cd project_wb
docker compose up -d
```


### Обращение к серверу

- Указанный ниже url соответствует случаю, когда сервер запущен на той же машине,
с которой производится обращение


#### Классификация одного изображения

- Указанный ниже путь к изображению является примером и должен быть заменен на
путь к изображению, которое вы хотите классифицировать
```python
import requests
import io
from PIL import Image


def image_to_byte_array(image: Image.Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


url = "http://localhost:8080/predict"
img = Image.open("test.jpg")
img_bytes = image_to_byte_array(img)
resp = requests.post(url, data=img_bytes).json()
print(resp['prob'], resp['verdict'])
```


#### Классификация нескольких изображений

- Подразумевается, что все изображения для классификации расположены в одной папке
- Указанный ниже путь к данной папке является примером и должен быть заменен на
путь к соответствующей папке на вашем компьютере
```python
import requests
import io
import os
from PIL import Image


def image_to_byte_array(image: Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


prep_data = {id: image_to_byte_array(Image.open(f"images/{id}")).hex() for id in os.listdir("images")}
url = "http://localhost:8080/batched_predict"
resp = requests.post(url, json=prep_data).json()

for id in resp.keys():
    print(f"{id}: {resp[id]['prob']:.3f} {resp[id]['verdict']}")
```
