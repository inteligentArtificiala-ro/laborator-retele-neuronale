"""Neuroni, straturi și o rețea cu mai multe straturi (MLP), construite din obiecte Value."""

import random

from engine import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, nin, activation="tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "tanh":
            return z.tanh()
        if self.activation == "relu":
            return z.relu()
        return z

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, activation="tanh"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, activation="tanh"):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1], activation if i < len(nouts) - 1 else "linear")
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
