"""Autoencoder PyTorch example.

Here we'll train an autoencoder based on the MNIST dataset.
"""

# Import
import pathlib
import time
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

import matplotlib.pyplot as plt


def main():
    # Determine if we're running on the CPU or GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')

    # Load training and testing data
    data_dir = pathlib.Path.home().joinpath('data', 'torch')
    train_data = torchvision.datasets.FashionMNIST(
        root=data_dir.as_posix(),
        train=True,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.FashionMNIST(
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

    # Move model to the GPU
    model.to(device)

    # Unpack test data and move to GPU
    X_test, y_test = zip(*test_data)
    X_test = torch.stack(X_test, dim=0)
    y_test = torch.tensor(y_test)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Simple training loop
    n_epochs = 100
    n_obs = train_dataloader.dataset.data.shape[0]
    train_loss = []
    test_loss = []
    test_accuracy = []
    start = time.time()
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch}\n------------------------------')
        epoch_loss = 0  # Total loss for the current epoch
        for batch, (X, y) in enumerate(train_dataloader):
            # Move data to the desired device
            X, y = X.to(device), y.to(device)

            # Forward pass, including loss calculation
            prediction = model(X)
            loss = criterion(prediction, y)
            epoch_loss += loss.item() / X.shape[0]  # Use average loss

            # Backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            if batch % 100 == 0:
                print(f'{batch * X.shape[0]:5d} / {n_obs:5d}, Loss: {loss.item():0.7f}')

        train_loss.append(epoch_loss)

        # -- End of epoch operations ---

        # Predict test data
        prediction_test = model(X_test)
        predicted_labels = torch.argmax(prediction_test, dim=1)
        test_loss_epoch = criterion(prediction_test, y_test)
        test_accuracy_epoch = torch.sum(predicted_labels == y_test) / y_test.shape[0]

        # Status update
        print(f'\nEpoch {epoch} complete')
        print(f'Test loss: {test_loss_epoch}')
        print(f'Test accuracy: {test_accuracy_epoch}\n')
        test_loss.append(test_loss_epoch.item())
        test_accuracy.append(test_accuracy_epoch.item())

    print(f'Training complete.')
    print(f'Elapsed time: {time.time() - start}')

    # Plot results
    save_dir = pathlib.Path.home().joinpath('results', 'torch', 'mnist')
    save_dir.mkdir(parents=True, exist_ok=True)

    fh, axh = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    ax = axh[0]
    ax.plot(train_loss, 'k')
    ax.set_ylabel('Training loss')
    ax.set_xlabel('Epoch')

    ax = axh[1]
    ax.plot(test_loss, 'k')
    ax.set_ylabel('Test loss')
    ax.set_xlabel('Epoch')

    ax = axh[2]
    ax.plot(test_accuracy, 'k')
    ax.set_ylabel('Test accuracy')
    ax.set_xlabel('Epoch')

    fh.tight_layout()
    fh.savefig(save_dir.joinpath('mnist_classification_ff.pdf'))

    # TODO:
    #   - Incorporate parameter saving
    #   - Look into train/test structure (see quickstart guide)
    #   - Consider tracking training and test accuracy for each epoch
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
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Linear(32, n_classes)  # Output layer
        # NOTE -- previously tried a slower compression step, but this seemed
        # to not train very quickly (likely due to the number of parameters)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output


if __name__ == '__main__':
    main()
