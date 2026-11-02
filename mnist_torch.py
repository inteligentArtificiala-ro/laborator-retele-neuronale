"""Același MLP pentru MNIST, în PyTorch."""

import time

import torch
import torch.nn as nn

from mnist_data import load_mnist


def make_model():
    return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def train(x_train, y_train, x_test, y_test, epochs=10, batch_size=64, lr=0.1, seed=0):
    torch.manual_seed(seed)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        start = time.time()
        order = torch.randperm(len(x_train))
        total_loss = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            loss = nn.functional.cross_entropy(model(x_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        with torch.no_grad():
            test_acc = (model(x_test).argmax(dim=1) == y_test).float().mean().item()
        print(
            f"epoca {epoch:2d}   pierdere {total_loss / len(x_train):.4f}   "
            f"acuratețe la testare {test_acc:.2%}   {time.time() - start:.1f} s"
        )
    return model


if __name__ == "__main__":
    data = load_mnist()
    start = time.time()
    train(*(torch.from_numpy(a) for a in data))
    print(f"Durata antrenării: {time.time() - start:.1f} s")
