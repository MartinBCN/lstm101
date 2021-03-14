import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os


def one_hot_encode(arr: np.array, n_labels: int) -> np.array:
    # Initialize the the encoded array)
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


def get_data(fn: str) -> np.array:
    """
    Load a text file, assign integers to characters and encode the whole text

    Returns
    -------

    """
    # open text file and read in data as `text`

    with open(fn, 'r') as f:
        text = f.read()

    # encode the text and map each character to an integer and vice versa

    # we create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to unique integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text])

    return encoded


def get_batches(arr: np.array, n_seqs: int, n_steps: int):
    """
    Create a generator that returns batches of size
       n_seqs x n_steps from arr.

    Parameters
    ----------
    arr: np.array
        Array you want to make batches from
    n_seqs: int
        Batch size, the number of sequences per batch
    n_steps: int
        Number of sequence steps per batch

    Returns
    -------
    Iterator of batches

    """

    # Get the number of characters per batch
    batch_size = n_seqs * n_steps

    # Get the number of batches we can make
    n_batches = len(arr) // batch_size

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]

    # Reshape into batch_size rows
    arr = arr.reshape(n_seqs, -1)

    # Make Batches
    for n in range(0, arr.shape[1], n_steps):
        end = n+n_steps
        # The features
        x = arr[:, n: end]
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]

        if end <= arr.shape[1]:
            y[:, -1] = arr[:, end]
        else:
            # start from the beginning?
            y[:, -1] = arr[:, 0]

        yield x, y


if __name__ == '__main__':
    data_dir = os.environ.get('DATA_DIR', 'data')
    fn = f'{data_dir}/anna.txt'
    anna = get_data(fn)

    print(anna[:100])
    print(len(anna))

    batches = get_batches(anna, 10, 50)
    x, y = next(batches)

    print('x\n', x[:10, :10])
    print('\ny\n', y[:10, :10])
