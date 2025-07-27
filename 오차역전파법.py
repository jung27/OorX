import math
import numbers

class Variable:
    __slots__ = ("data", "_grad", "_parents", "_op", "_backward")

    def __init__(self, data, parents=(), op=""):
        if not isinstance(data, numbers.Number):
            raise TypeError(f"Unsupported type {type(data)} for Variable(data)")
        self.data = float(data)
        self._grad = 0.0
        self._parents = tuple(parents)
        self._op = op
        self._backward = lambda: None

    @property
    def grad(self):
        return self._grad

    def zero_grad(self):
        # clear gradients for entire graph
        for node in self._build_topo():
            node._grad = 0.0

    def _build_topo(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents:
                    build(p)
                topo.append(v)
        build(self)
        return topo

    def backward(self):
        topo = self._build_topo()
        # reset gradients
        for node in topo:
            node._grad = 0.0
        # seed
        self._grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, (self, other), '+')
        def _backward():
            self._grad += out._grad
            other._grad += out._grad
        out._backward = _backward
        return out
    __radd__ = __add__

    def __neg__(self):
        out = Variable(-self.data, (self,), 'neg')
        def _backward():
            self._grad -= out._grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other if isinstance(other, Variable) else Variable(other)).__neg__()
    def __rsub__(self, other):
        return (other if isinstance(other, Variable) else Variable(other)) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, (self, other), '*')
        def _backward():
            self._grad += other.data * out._grad
            other._grad += self.data * out._grad
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data / other.data, (self, other), '/')
        def _backward():
            self._grad += (1 / other.data) * out._grad
            other._grad -= (self.data / (other.data ** 2)) * out._grad
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(other.data / self.data, (other, self), 'r/')
        def _backward():
            other._grad += (1 / self.data) * out._grad
            self._grad -= (other.data / (self.data ** 2)) * out._grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        if isinstance(exponent, Variable):
            out = Variable(self.data ** exponent.data, (self, exponent), '**')
            def _backward():
                if self.data <= 0:
                    raise ValueError("Base must be > 0 for variable exponent differentiation")
                self._grad += exponent.data * (self.data ** (exponent.data - 1)) * out._grad
                exponent._grad += out.data * math.log(self.data) * out._grad
            out._backward = _backward
        else:
            out = Variable(self.data ** exponent, (self,), f'**{exponent}')
            def _backward():
                self._grad += exponent * (self.data ** (exponent - 1)) * out._grad
            out._backward = _backward
        return out

    def __rpow__(self, base):
        if not isinstance(base, numbers.Number):
            return Variable(base) ** self
        if base <= 0:
            raise ValueError("Base must be > 0 for exponentiation differentiation")
        out = Variable(base ** self.data, (self,), f'{base}**x')
        def _backward():
            self._grad += out.data * math.log(base) * out._grad
        out._backward = _backward
        return out

    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, Variable) else other)

    def __repr__(self):
        return f"Variable({self.data})"