import warnings
import argparse
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


class ViT:
    """
    Класс для работы с моделью
    """
    def __init__(self):
        """
        Конструктор
        """
        self.model_name = "EGORsGOR/vit-spam"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        warnings.filterwarnings('ignore')
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTForImageClassification.from_pretrained(self.model_name).to(self.device)
        warnings.filterwarnings('default')
        self.soft_max = torch.nn.functional.softmax

    def predict(self, image: Image.Image) -> torch.Tensor:
        """
        Метод предсказывает вероятности классов переданного изображения
        :param image: Изображение для классификации
        :return: torch.Tensor с вероятностями 0-го и 1-го классов соответственно
        """
        feature = self.feature_extractor(image, return_tensors='pt').to(self.device)
        with torch.no_grad():
            res = self.model(**feature).logits
        probabilities = self.soft_max(res, dim=1)

        return probabilities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image to classify")
    args = parser.parse_args()

    model = ViT()
    img = Image.open(args.image)
    probs = model.predict(img)
    labels = {0: "NON-SPAM", 1: "SPAM"}

    print(f"predicted class is {labels[probs.argmax().item()]}")
    print(f"spam probability = {round(probs[1].item(), 3)}")
