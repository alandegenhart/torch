"""Autoencoder PyTorch example.

Here we'll train an autoencoder based on the MNIST dataset.
"""

# Import
import pathlib
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms


def main():
    # Load training and testing data
    data_dir = pathlib.Path.home().joinpath('data', 'torch')
    train_data = torchvision.datasets.MNIST(
        root=data_dir.as_posix(),
        train=True,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.MNIST(
        root=data_dir.as_posix(),
        train=False,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )

    # The structure of the dataset consists of tuples
    X, y = train_data[0]
    print(f'Shape of image data: {X.shape}')

    # Create Dataloader objects
    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Create network
    model = FFNet(input_size=X.numel(), n_classes=10)
    print(model)

    # Define loss function and optimizer
    # CrossEntropyLoss takes a C-element vector and a target class as input.
    # Per the PyTorch documentation, other options are possible, though not
    # recommended.  The input vector should also be un-normalized -- I take this
    # to mean that there should be no activation function preceding the output.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Simple training loop
    n_epochs = 2
    for epoch in range(n_epochs):
        for batch, (X, y) in enumerate(train_dataloader):
            # Forward pass, including loss calculation
            prediction = model(X)
            loss = criterion(prediction, y)

            # Backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # TODO:
    #   - Finish training loop
    #   - Keep track of loss (save for plotting later)
    #   - At the end of each training loop calculate the test loss
    #   - Calculate final classification accuracy
    #   - Look @ documentation to see how batches are handled -- for example, in
    #       the above code, X has shape [batch_size, 1, 28, 28], but the network
    #       is able to handle this fine.
    #   - Once the feed-forward network is working, create the autoencoder
    #       network and test.
    #   - Could look into transfer learning -- take the weights from the
    #       classification model, fix the parameters, and then train the
    #       decoder model.  Could also initialize the parameters of the encoder
    #       model based on the classification model parameters.
    
    return None


class FFNet(torch.nn.Module):
    """Linear autoencoder network."""
    def __init__(self, input_size, n_classes):
        super(FFNet, self).__init__()
        self.input_size = input_size

        # Define network layers
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Linear(16, n_classes)  # Output layer

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output


if __name__ == '__main__':
    main()
