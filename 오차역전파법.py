# ===== 오차역전파법.py =====
import math, numbers

class Variable:
    __slots__ = ("data","_grad","_parents","_op","_backward")
    def __init__(self,data,parents=(),op=""):
        if not isinstance(data,numbers.Number):
            raise TypeError(f"Unsupported type {type(data)}")
        self.data = float(data)
        self._grad = 0.0
        self._parents = tuple(parents)
        self._op = op
        self._backward = lambda: None

    @property
    def grad(self): return self._grad

    def zero_grad(self):
        for n in self._build_topo(): n._grad = 0.0

    def _build_topo(self):
        topo,vis = [],set()
        def build(v):
            if v not in vis:
                vis.add(v)
                for p in v._parents: build(p)
                topo.append(v)
        build(self)
        return topo

    def backward(self):
        topo = self._build_topo()
        for n in topo: n._grad = 0.0
        self._grad = 1.0
        for n in reversed(topo): n._backward()

    def __add__(self,other):
        other = other if isinstance(other,Variable) else Variable(other)
        out = Variable(self.data+other.data,(self,other),'+')
        def _back():
            self._grad  += out._grad
            other._grad += out._grad
        out._backward = _back
        return out
    __radd__ = __add__

    def __neg__(self):
        out = Variable(-self.data,(self,),'neg')
        def _back(): self._grad -= out._grad
        out._backward = _back
        return out

    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return Variable(other)+(-self)

    def __mul__(self,other):
        other = other if isinstance(other,Variable) else Variable(other)
        out = Variable(self.data*other.data,(self,other),'*')
        def _back():
            self._grad  += other.data * out._grad
            other._grad += self.data      * out._grad
        out._backward = _back
        return out
    __rmul__ = __mul__

    def __truediv__(self,other):
        other = other if isinstance(other,Variable) else Variable(other)
        out = Variable(self.data/other.data,(self,other),'/')
        def _back():
            self._grad  += (1/other.data)*out._grad
            other._grad -= (self.data/(other.data**2))*out._grad
        out._backward = _back
        return out
    def __rtruediv__(self,other): return Variable(other)/self

    def __pow__(self,exp):
        if isinstance(exp,Variable):
            out = Variable(self.data**exp.data,(self,exp),'**')
            def _back():
                self._grad += exp.data*(self.data**(exp.data-1))*out._grad
                exp._grad += out.data*math.log(self.data)*out._grad
            out._backward = _back
        else:
            out = Variable(self.data**exp,(self,),f'**{exp}')
            def _back(): self._grad += exp*(self.data**(exp-1))*out._grad
            out._backward = _back
        return out
    def __rpow__(self,base): return Variable(base)**self

    def relu(self):
        val = max(0.0, self.data)
        out = Variable(val,(self,),'relu')
        def _back(): self._grad += (val>0)*out._grad
        out._backward = _back
        return out

    def sigmoid(self):
        val = 1/(1+math.exp(-self.data))
        out = Variable(val,(self,),'sigmoid')
        def _back(): self._grad += val*(1-val)*out._grad
        out._backward = _back
        return out

    def log(self):
        out = Variable(math.log(self.data),(self,),'log')
        def _back(): self._grad += (1/self.data)*out._grad
        out._backward = _back
        return out

    def __repr__(self): return f"Variable({self.data:.4f})"