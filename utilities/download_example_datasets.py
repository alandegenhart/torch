"""Download datasets."""

# Import
import pathlib
from torchvision import datasets

# Download training data from open datasets
data_dir = pathlib.Path.home().joinpath('data', 'torch')
training_data = datasets.MNIST(
    root=data_dir.as_posix(),
    train=False,
    download=False,
)

print(training_data)