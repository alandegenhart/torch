"""PyTorch tutorial 2: datasets and dataloaders."""

# Imports
import pathlib

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


def main():
    # Define directories
    data_dir = pathlib.Path.home().joinpath('data', 'torch')
    results_dir = pathlib.Path.home().joinpath('results', 'torch',
                                               '2_datasets_dataloaders')
    results_dir.mkdir(exist_ok=True, parents=True)
    fig_prefix = '2_datasets_dataloaders'

    # Download data
    download = False  # Data should already be downloaded
    
    # Training data
    training_data = datasets.FashionMNIST(
        root=data_dir.as_posix(),
        train=True,
        download=download,
        transform=ToTensor()
    )

    # Test data
    test_data = datasets.FashionMNIST(
        root=data_dir.as_posix(),
        train=False,
        download=download,
        transform=ToTensor()
    )

    # Define mapping from labels to items
    labels_map = {
        0: 'T-Shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot',
    }

    # Plot some sample images
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):  # subplots are 1-indexed
        # Get a random image
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]

        # Plot
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')

    plt.savefig(results_dir.joinpath(f'{fig_prefix}_sample_images.png'))

    # Create dataloaders to illustrate iterating through datasets
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Iterate over dataloader
    figure = plt.figure(figsize=(3, 3))
    train_features, train_labels = next(iter(train_dataloader))
    print(f'Features batch shape: {train_features.size()}')
    print(f'Labels batch shape: {train_labels.size()}')
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap='gray')
    plt.title(labels_map[label.item()])
    plt.savefig(results_dir.joinpath(f'{fig_prefix}_dataloader_img.png'))

    return None


if __name__ == '__main__':
    main()

