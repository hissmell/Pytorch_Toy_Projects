from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torchvision.datasets import ImageFolder,MNIST
from torch.utils.data import DataLoader, Dataset

import os

def load_mnist_dataset(means=(0.5), stds=(0.5),batch_size=64, num_workers=0):
    image_transformation = Compose([ToTensor(), Normalize(means, stds)])
    train_dataset = MNIST(root='C:\\Users\\82102\\PycharmProjects\\ReinforcementLearning\\Toy_project\\VAE_MNIST\\data'
                          ,train=True
                          ,transform=image_transformation
                          ,download=True)
    valid_dataset = MNIST(root='C:\\Users\\82102\\PycharmProjects\\ReinforcementLearning\\Toy_project\\VAE_MNIST\\data'
                          ,train=False
                          ,transform=image_transformation
                          ,download=True)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_data_loader,valid_data_loader

if __name__ == '__main__':
    train_data_loader,valid_data_loader = load_mnist_dataset()
    images,labels = next(iter(train_data_loader))
    print(images.size())
    print(images.dtype)
    print(labels.size())
    print(labels.dtype)