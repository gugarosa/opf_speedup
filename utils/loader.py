import numpy as np
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s


def load_dataset(file_path, normalize=True):
    """Loads data from a .txt file and parses it.

    Args:
        file_path (str): Input file to be loaded.
        normalize (bool): Whether data should be normalized or not.

    Returns:
        Samples and labels arrays.

    """

    # Loading a .txt file to a numpy array
    txt = l.load_txt(file_path)

    # Parsing a pre-loaded numpy array
    X, Y = p.parse_loader(txt)

    # Checks if data should be normalized
    if normalize:
        # Normalizes the data
        X = (X - np.min(X)) / np.ptp(X)

    return X, Y


def load_split_dataset(file_path, train_split=0.5, normalize=True, random_state=1):
    """Loads data from a .txt file, parses it and splits into training and validation sets.

    Args:
        file_path (str): Input file to be loaded.
        train_split (float): Percentage of training set.
        normalize (bool): Whether data should be normalized or not.
        random_state (int): Seed used to provide a deterministic trait.

    Returns:
        Training and validation sets along their indexes.

    """

    # Loading a .txt file to a numpy array
    txt = l.load_txt(file_path)

    # Parsing a pre-loaded numpy array
    X, Y = p.parse_loader(txt)

    # Checks if data should be normalized
    if normalize:
        # Normalizes the data
        X = (X - np.min(X)) / np.ptp(X)

    # Splitting data into training and validation sets
    X_train, X_val, Y_train, Y_val = s.split(X, Y, percentage=train_split, random_state=random_state)

    return X_train, Y_train, X_val, Y_val
