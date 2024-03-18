import torch
from classification import MammographyClassifier
from model import MammographyModel
from sklearn.metrics import accuracy_score
import numpy as np


if __name__ == "__main__":
    model_path = 'mammography_model.pth'
    num_classes = 2
    model = MammographyModel(num_classes=num_classes)
    classifier = MammographyClassifier(model_path, num_classes)
    image_path = ""
    test_label = ""
    predicted_class, probabilities = classifier.classify_image(image_path)
    max_prob, predicted_class_index = torch.max(probabilities, dim=1)
    max_prob = max_prob.item()
    print(max_prob)

    class_names = ['', 'd']
    predicted_class_name = class_names[predicted_class]
    print(f'The predicted class is: {predicted_class_name}')
    model.display_image_with_prediction(image_path, predicted_class_name)
    print(f'Probabilities: {probabilities}')
    
    

