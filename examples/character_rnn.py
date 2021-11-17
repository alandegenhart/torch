"""NLP from scratch: classifying names with a character-level RNN

Adapted from the tutorial on PyTorch (see pytorch.org/tutorials).
"""

# Imports
import pathlib
import string
import unicodedata
import random
import time
import math

import matplotlib.pyplot as plt

import torch
import torch.nn

# Define a list of possible letters (might be better to come up with a better
# way to do this that avoids using a global variable)
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def prepare_data():
    """Prepare data for training."""
    data_dir = pathlib.Path.home().joinpath(
        'data', 'torch', 'character_rnn', 'names'
    )
    print(list(data_dir.glob('*.txt')))

    def unicode_to_ascii(s):
        """Convert unicode string to plain ASCII.
        
        Source: stackoverflow.com/a/518232/2809427
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

    # Test that unicode conversion is working
    print(unicode_to_ascii('Ślusàrski'))

    # Build a dictionary with a list of names per language
    category_lines = {}
    all_categories = []

    def read_lines(filepath: pathlib.Path):
        """Open a file and convert to a list of strings, 1 per line."""
        with filepath.open(encoding='utf-8') as f:
            # Read all lines, remove whitespace, and split by new line character
            s = f.read().strip().split('\n')

        return [unicode_to_ascii(line) for line in s]

    # Iterate over all files
    for filepath in data_dir.glob('*.txt'):
        # Get language name from filename
        language = filepath.stem
        all_categories.append(language)

        # Read names
        lines = read_lines(filepath)
        category_lines[language] = lines

    n_categories = len(all_categories)
    print(f'Number of categories: {n_categories}')
    idx = 5
    ex_cat_name = all_categories[idx]
    print(f'Examples names ({ex_cat_name}): {category_lines[ex_cat_name][:5]}')

    return all_categories, category_lines


def letter_to_index(letter):
    """Get index encoding for a single letter."""
    return all_letters.find(letter)


def letter_to_tensor(letter):
    """Example of a one-hot encoding."""
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    """Convert a sequence of letters to a tensor."""
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li, 0, letter_to_index(letter)] = 1
    return tensor


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)  # Input to output
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input_state, hidden_state):
        combined = torch.cat((input_state, hidden_state), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        """Initialize hidden state."""
        return torch.zeros(1, self.hidden_size, device=next(self.parameters()).device)


def output_to_category(output_state):
    """Converts likelihood tensor to category."""
    top_n, top_i = output_state.topk(1)  # Get largest value and associated index
    category_i = top_i[0].item()  # Get index for greatest likelihood
    category = all_categories[category_i]
    return category, category_i  # Return both the category and associated index


def random_choice(x):
    """Choose a random element from an input tensor."""
    return x[random.randint(0, len(x) - 1)]


def random_training_example(all_categories, category_lines):
    """Get a random training data example."""
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)  # Get index for category
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def train(rnn, loss_fn, category_tensor, line_tensor, learning_rate=0.005):
    """Train network."""
    hidden_state = rnn.init_hidden().to(line_tensor.device)
    rnn.zero_grad()  # Initialize the hidden state

    # Predict the output -- in this case we only care about the final prediction
    # after observing all of the letters in the line.  Also note that the
    # hidden state gets passed back into the RNN after each iteration.
    for i in range(line_tensor.size()[0]):
        output_state, hidden_state = rnn(line_tensor[i], hidden_state)

    # Calculate loss and run backward pass to compute gradients
    loss = loss_fn(output_state, category_tensor)
    loss.backward()

    # Update parameters
    for p in rnn.parameters():
        # learning rule: p = p - d_p * lr
        # Note that 'add_' specifies an in-place add operation, with optional
        # input 'alpha' a scale factor.
        p.data.add_(p.grad.data, alpha=-learning_rate)

        # NOTE -- this code seems to be a bit outdated.  Using p.data here
        # allows the parameters to be changed without autograd tracking the
        # changes.  It seems that newer code should use torch.no_grad():
        #
        # with torch.no_grad():
        #   for p in rnn.parameters():
        #       p -= learning_rate * p.grad

    return output_state, loss


def convert_all_data(all_categories, category_lines, device):
    """Pre-compute tensor representation for all data"""
    category_tensor = []
    line_tensor = []
    all_lines = []
    for category in all_categories:
        for line in category_lines[category]:
            category_tensor.append(torch.tensor([all_categories.index(category)], dtype=torch.long).to(device))
            line_tensor.append(line_to_tensor(line).to(device))
            all_lines.append(line)

    return category_tensor, line_tensor, all_lines


def elapsed_time(start):
    """Returns time elapsed since start as a string."""
    now = time.time()
    elapsed_s = now - start
    elapsed_min = math.floor(elapsed_s / 60)
    elapsed_s = elapsed_s % 60
    return f'{elapsed_min}m {elapsed_s}s'


def evaluate(rnn, line_tensor, device):
    """Predict category for a single line."""
    hidden_state = rnn.init_hidden().to(device)
    for i in range(line_tensor.size()[0]):
        output_state, hidden_state = rnn(line_tensor[i], hidden_state)
    return output_state


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device} device.')

    # Load data
    all_categories, category_lines = prepare_data()

    # Train network
    print('\n\n--- Training network --- \n\n')

    # Initialize network
    n_hidden = 128
    n_categories = len(all_categories)
    rnn = RNN(n_letters, n_hidden, n_categories)
    rnn.to(device)

    # Prepare data
    category_tensor, line_tensor, all_lines = convert_all_data(all_categories, category_lines, device)
    n_obs = len(category_tensor)

    # Specify training parameters
    n_iter = 100000
    print_every = 5000
    plot_every = 1000

    # Initialize loss
    learning_rate = 0.005
    loss_fn = torch.nn.NLLLoss()  # Negative log likelihood loss -- expects log-probabilities as input
    current_loss = torch.tensor(0.0, device=device)  # Average loss -- want this to live on the GPU if possible
    all_losses = []

    start_time = time.time()
    for i in range(1, n_iter + 1):
        # Get a random training example and take a single training step
        idx = random.randint(0, n_obs-1)
        output_state, loss = train(rnn, loss_fn, category_tensor[idx], line_tensor[idx], learning_rate=learning_rate)
        current_loss += loss  # Keep track of overall loss

        # Print status
        if i % print_every == 0:
            predicted_category, predicted_index = output_to_category(output_state)
            actual_category = all_categories[category_tensor[idx].item()]
            prediction_str = f'{loss.item()} {all_lines[idx]} / {predicted_category} '
            if predicted_category == actual_category:
                prediction_str += '✓'
            else:
                prediction_str += f'✗ ({actual_category})'
            print(f'{i} {100 * i/n_iter}% ({elapsed_time(start_time)}) - ' + prediction_str)

        # Keep track of loss
        if i % plot_every == 0:
            avg_loss = current_loss.item() / plot_every
            all_losses.append(avg_loss)
            current_loss = 0  # Reset loss accumulator

    # Plot loss
    plt.figure()
    plt.plot(all_losses, 'k')
    plt.xlabel(f'Iter / {plot_every}')
    plt.ylabel('Average loss')
    plt.title('Training loss')
    plt.show()

    # --- Evaluate network ---
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(n_categories, n_categories)
    n_samples = 10000

    # Randomly sample data and populate confusion matrix
    for i in range(n_samples):
        idx = random.randint(0, n_obs - 1)
        output_state = evaluate(rnn, line_tensor[idx], device)
        predicted_category, predicted_index = output_to_category(output_state)
        category_i = category_tensor[idx].item()  # Actual index
        confusion_matrix[category_i, predicted_index] += 1

    # Normalize confusion matrix
    row_sum = torch.sum(confusion_matrix, 1, keepdim=True)  # rows x 1
    confusion_matrix_norm = confusion_matrix / torch.tile(row_sum, (1, n_categories))

    # Plot confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix_norm.numpy())
    fig.colorbar(cax)

    # Add axis ticks
    ax.set_xticklabels(all_categories, rotation=90)
    ax.set_yticklabels(all_categories)
    ax.set_xticks(range(n_categories))
    ax.set_yticks(range(n_categories))

    plt.show()

    # NOTES:
    # - Training is currently slower when using the GPU.  Not sure if this is
    #   due to GPU overhead or not.  I don't think there are unnecessary
    #   transfers from the GPU to the CPU (and vice-versa), but it might be
    #   worth checking at some point to ensure we get the gpu speedup when
    #   possible.
    # - Random sampling needs to be improved.  Right now data is just selected
    #   randomly from all available samples.  This means that there will be
    #   biases in the selection process towards classes with more observations.
    #   This should be fixed at some point.
