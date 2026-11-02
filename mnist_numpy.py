"""Un MLP 784 → 128 → 10 pentru MNIST, cu NumPy și retropropagarea scrisă de mână."""


import numpy as np

from mnist_data import load_mnist


def init_params(n_in=784, n_hidden=128, n_out=10, seed=0):
    rng = np.random.default_rng(seed)
    # Inițializarea He: varianța 2 / numărul de intrări, potrivită pentru ReLU.
    return {
        "W1": (rng.standard_normal((n_in, n_hidden)) * np.sqrt(2 / n_in)).astype(np.float32),
        "b1": np.zeros(n_hidden, dtype=np.float32),
        "W2": (rng.standard_normal((n_hidden, n_out)) * np.sqrt(2 / n_hidden)).astype(np.float32),
        "b2": np.zeros(n_out, dtype=np.float32),
    }


def forward(params, x):
    z1 = x @ params["W1"] + params["b1"]
    h = np.maximum(z1, 0)
    logits = h @ params["W2"] + params["b2"]
    return logits, (x, z1, h)


def softmax(logits):
    # Fără maxim, exp poate depăși cel mai mare număr reprezentabil; rezultatul este același.
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
