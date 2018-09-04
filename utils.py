import numpy as np

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def train_test_split(X, Y, test_size):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    total_examples = X.shape[0]
    test_examples = int(total_examples * test_size)
    train_examples = total_examples - test_examples

    x_train = X[:train_examples] # First Z examples
    y_train = Y[:train_examples]
    x_test = X[-test_examples:] # Last Z examples
    y_test = Y[-test_examples:]

    return x_train, x_test, y_train, y_test