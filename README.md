# Wildberries project 

Данный репозиторий представляет собой решение 5-го этапа дипломного проекта Техношколы Data Scientist WB по направлению 'Репутация пользователей'

## Инструкция по сборке репозитория 
```commandline
git clone https://github.com/egorSgor1/project_wb.git
docker compose up -d
```

## Инструкция по обращению к серверу

### Для классификации одного изображения

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

### Для классификации нескольких изображений

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
