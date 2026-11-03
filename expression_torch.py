"""Aceeași expresie în PyTorch, graful lui autograd și torch.autograd.gradcheck."""

import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(7.0, requires_grad=True)
f = torch.tensor(-2.0, requires_grad=True)
L = (a * b + c) * f
print("L =", L.item())

print("operația:", L.grad_fn.name())
print("calculat din:", [type(fn).__name__ for fn, _ in L.grad_fn.next_functions])

L.backward()
print("gradienții lui a, b, c, f:", a.grad.item(), b.grad.item(), c.grad.item(), f.grad.item())


def neuron(x, w, b):
    return torch.tanh(x @ w + b)


x = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
w = torch.tensor([0.5, 0.25], dtype=torch.float64, requires_grad=True)
b = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
print("gradcheck:", torch.autograd.gradcheck(neuron, (x, w, b)))
