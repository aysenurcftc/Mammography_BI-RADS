from PIL import Image
from torchvision import transforms

class MammographyProcessor:
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  
        return input_batch