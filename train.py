import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MammographyModel
from processor import MammographyProcessor


def main():
    # Define data directories
    data_dir = 'data/'
    
    # Initialize MammographyProcessor
    processor = MammographyProcessor()

    # Define datasets and dataloaders
    image_datasets = {x: datasets.ImageFolder(f'{data_dir}/{x}', processor.preprocess) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    # Initialize model
    model = MammographyModel(num_classes=len(class_names))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train(dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10)

    # Save the trained model
    model.save('mammography_model.pth')


if __name__ == "__main__":
    main()