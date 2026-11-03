"""Expresia L = (a * b + c) * f și gradienții lui L."""

from engine import Value

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
print(L)

L.backward()
for v in [L, f, d, c, e, b, a]:
    print(f"dL/d{v.label} = {v.grad}")
