import io
from flask import Flask, request, jsonify
from PIL import Image
import ViT

app = Flask(__name__, static_url_path="")
model = ViT.ViT()


@app.route("/predict", methods=['POST'])
def predict():
    """
    Функция реализует ответ сервера на POST-запрос, полученный по пути "/predict".
    В рамках данного запроса сервер получает изображение в виде набора байт,
    обрабатывает его, прогоняет через модель и возвращает словарь с вероятностью
    отнесения изображения к спаму и предсказанным классом.
    """
    img_bytes = request.get_data()
    img = Image.open(io.BytesIO(img_bytes))
    probs = model.predict(img)[0]

    return jsonify({"prob": round(probs[1].item(), 3), "verdict": probs.argmax().item()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
