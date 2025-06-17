import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loader(name='cifar10', batch_size=128, num_workers=4):
    if name.lower() == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        test = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    elif name.lower() == 'tinyimagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
        train = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform=train_transform)
        test = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
