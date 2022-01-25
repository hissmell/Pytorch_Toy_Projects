from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

import os


def load_image_dataset(data_dir_path=None, image_size=(64, 64), means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5),
                       batch_size=64, num_workers=1):
    if type(image_size) == int:
        image_size = (image_size, image_size)
    if data_dir_path == None:
        data_dir_path = os.getcwd()

    image_transformation = Compose([Resize(image_size), ToTensor(), Normalize(means, stds)])
    train_dataset = ImageFolder(os.path.join(data_dir_path, 'train'), transform=image_transformation)
    valid_dataset = ImageFolder(os.path.join(data_dir_path, 'valid'), transform=image_transformation)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_data_loader, valid_data_loader