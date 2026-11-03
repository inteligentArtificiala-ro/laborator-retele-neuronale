"""Setul de date MNIST: 70.000 de cifre scrise de mână, de 28 × 28 pixeli."""

import hashlib
import os
import urllib.request

import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def download(path="data/mnist.npz"):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Descărcăm {URL}")
        urllib.request.urlretrieve(URL, path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{path} nu are amprenta așteptată. Ștergeți fișierul și rulați din nou.")
    return path


def load_mnist(path="data/mnist.npz"):
    with np.load(download(path)) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255
    return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Antrenare: {x_train.shape}, testare: {x_test.shape}")
    print("Cifre pe clasă în setul de antrenare:", np.bincount(y_train).tolist())
