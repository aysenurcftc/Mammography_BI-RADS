

from model import MammographyModel
from processor import MammographyProcessor


class MammographyClassifier:
    def __init__(self, model_path, num_classes):
        self.model = MammographyModel(num_classes)
        self.model.load(model_path)
        self.image_processor = MammographyProcessor()

    def classify_image(self, image_path):
        input_batch = self.image_processor.preprocess_image(image_path)
        output = self.model.predict(input_batch)
        return output