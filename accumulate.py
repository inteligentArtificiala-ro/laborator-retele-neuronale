"""De ce gradienții se adună: în y = x * x + x, x este folosit de trei ori."""

from engine import Value

x = Value(3.0)
y = x * x + x
y.backward()
print("dy/dx =", x.grad)

y.backward()
print("după încă un backward():", x.grad)
