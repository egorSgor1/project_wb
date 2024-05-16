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


@app.route("/batched_predict", methods=['POST'])
def predict_batch():
    """
    Функция реализует ответ сервера на POST-запрос, полученный по пути "/batched_predict".
    В рамках данного запроса сервер получает словарь с изображениями в байтовом виде, обрабатывает их,
    делит на батчи фиксированного размера, затем прогоняет эти батчи через модель
    и возвращает словарь с вероятностью спама и предсказанным классом для каждого изображения.
    """
    data = request.get_json(force=True)
    img_bytes = list(map(lambda x: Image.open(io.BytesIO(bytes.fromhex(x))), data.values()))
    batch_size = 32
    num_batches = len(img_bytes) // batch_size
    num_batches += 1 if (len(img_bytes) % batch_size != 0) else 0
    batches = [img_bytes[i*batch_size:min(len(img_bytes), (i + 1)*batch_size)] for i in range(num_batches)]
    res = {}
    for i, batch in enumerate(batches):
        probs = model.predict(batch)
        cur_ids = list(data.keys())[i*batch_size:min(len(img_bytes), (i + 1)*batch_size)]
        for j, id in enumerate(cur_ids):
            res[id] = {"prob": round(probs[j][1].item(), 3), "verdict": probs[j].argmax().item()}

    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
