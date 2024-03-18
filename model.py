import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class MammographyModel:
    
    def __init__(self, num_classes):
            
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.best_loss = float('inf')  
        self.patience = 5  
        self.counter = 0  
        self.early_stop = False  

    def train(self, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=2):
        self.model = self.model.to(device)
        for epoch in range(num_epochs):
            if not self.early_stop:  
                for phase in ['train', 'test']:
                    if phase == 'train':
                        self.model.train()
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                 
                    if phase == 'test':
                        if epoch_loss < self.best_loss:
                            self.best_loss = epoch_loss
                            self.counter = 0
                        else:
                            self.counter += 1
                            if self.counter >= self.patience:
                                self.early_stop = True
                                print("Early stopping activated!")
                                break

        print("Training complete!")
        
    def predict(self, input_batch):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = self.softmax(outputs)
            _, predicted = torch.max(outputs, 1)
        return predicted, probabilities
    
    
    def display_image_with_prediction(self, image_path, predicted_class_name):
        image = Image.open(image_path)
        image = np.array(image)

        plt.imshow(image)
        plt.axis('off')
        plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')
        plt.show()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

