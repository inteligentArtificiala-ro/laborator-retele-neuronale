"""Diagramele grafurilor computaționale, salvate în directorul imagini/."""

import os

from engine import Value
from graph import draw_graph

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

    print("Diagramele au fost salvate în", OUT)
