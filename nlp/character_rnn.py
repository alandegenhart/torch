"""NLP from scratch: classifying names with a character-level RNN

Adapted from the tutorial on PyTorch (see pytorch.org/tutorials).
"""

# Imports
import pathlib
import string
import unicodedata
import io


def prepare_data():
    """Prepare data for training."""
    data_dir = pathlib.Path.home().joinpath('data', 'torch', 'character_rnn',
                                            'names')
    print(list(data_dir.glob('*.txt')))

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

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


if __name__ == '__main__':
    all_categories, category_lines = prepare_data()
