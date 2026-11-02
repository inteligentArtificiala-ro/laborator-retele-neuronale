"""Diagramele grafurilor computaționale, salvate în directorul imagini/."""

import os

from engine import Value
from graph import draw_graph
from nn import MLP

OUT = "imagini"


def expression():
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(7.0, label="c")
    f = Value(-2.0, label="f")
    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"
    L = d * f
    L.label = "L"
    return L


def neuron():
    x1 = Value(1.0, label="x1")
    x2 = Value(-2.0, label="x2")
    w1 = Value(0.5, label="w1")
    w2 = Value(0.25, label="w2")
    b = Value(0.8, label="b")
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    s = x1w1 + x2w2
    s.label = "s"
    n = s + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"
    return o


def tiny_mlp():
    model = MLP(2, [2, 1], activation="relu")
    hidden, output = model.layers[0].neurons, model.layers[1].neurons[0]
    values = {"w11": 0.5, "w12": -0.25, "b1": 0.1, "w21": -0.4, "w22": 0.3, "b2": 0.2}
    for j, n in enumerate(hidden, start=1):
        for i, w in enumerate(n.w, start=1):
            w.label, w.data = f"w{j}{i}", values[f"w{j}{i}"]
        n.b.label, n.b.data = f"b{j}", values[f"b{j}"]
    for j, (w, value) in enumerate(zip(output.w, [1.2, 0.6]), start=1):
        w.label, w.data = f"v{j}", value
    output.b.label, output.b.data = "c", 0.3

    x = [Value(1.0, label="x1"), Value(-1.0, label="x2")]
    y = model(x)
    y.label = "y"
    loss = (y - 1.0) ** 2
    loss.label = "pierdere"
    return loss


def draw_before_and_after():
    L = expression()
    draw_graph(L, f"{OUT}/expresie-inainte.svg")
    L.backward()
    draw_graph(L, f"{OUT}/expresie-dupa.svg")
    return L


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    L = draw_before_and_after()
    print("Expresia L = (a * b + c) * f")
    print(f"  L = {L.data}")

    o = neuron()
    o.backward()
    draw_graph(o, f"{OUT}/neuron.svg")
    print(f"Neuronul: o = {o.data:.4f}")

    loss = tiny_mlp()
    loss.backward()
    draw_graph(loss, f"{OUT}/retea-mica.svg")
    print(f"Rețeaua mică: pierderea = {loss.data:.4f}")

    print("Diagramele au fost salvate în", OUT)
