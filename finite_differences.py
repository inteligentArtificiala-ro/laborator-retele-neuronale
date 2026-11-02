"""Gradienții unui neuron, verificați cu diferențe finite."""

from engine import Value


def neuron(x1, x2, w1, w2, b, tanh_from_exp=False):
    n = x1 * w1 + x2 * w2 + b
    if tanh_from_exp:
        e = (2 * n).exp()
        return (e - 1) / (e + 1)
    return n.tanh()


inputs = {"x1": 1.0, "x2": -2.0, "w1": 0.5, "w2": 0.25, "b": 0.8}
eps = 1e-6

for tanh_from_exp in [False, True]:
    values = {k: Value(v) for k, v in inputs.items()}
    o = neuron(**values, tanh_from_exp=tanh_from_exp)
    o.backward()
    print("tanh din exponențiale" if tanh_from_exp else "tanh direct", f"o = {o.data:.6f}")
    for k in inputs:
        up = {n: Value(v + eps * (n == k)) for n, v in inputs.items()}
        down = {n: Value(v - eps * (n == k)) for n, v in inputs.items()}
        numeric = (neuron(**up).data - neuron(**down).data) / (2 * eps)
        print(f"  do/d{k}: backward {values[k].grad:+.6f}, diferențe finite {numeric:+.6f}")
