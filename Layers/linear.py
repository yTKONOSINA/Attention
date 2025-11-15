from tensor import Tensor
import random

class Linear:
    def __init__(self,
                  m : int,
                  n : int,
                  w : list = None,
                  b : list = None):
        self.m = m
        self.n = n
        if w:
            self.w = Tensor(w)
        else:
            self.w = Tensor([[random.random() for _ in range(n)]
                        for _ in range(m)])
        if b:
            self.b = Tensor(b)
        else:
            self.b = Tensor([random.random() for _ in range(n)])
    
    def forward(self, x : Tensor) -> Tensor:
        res = x @ self.w
        
        def add_bias(data, bias):
            if not isinstance(data, list):
                return data + bias
            return [add_bias(x, bias) for x in data]
            
        assert len(res.shape) == len(self.b.shape), "Bias shape must match result shape"
        res.tensor = add_bias(res.tensor, self.b.tensor)
        return res

if __name__ == "__main__":
    layer = Linear(3, 54, bias = 1, data = 1.0)
    x = Tensor([[3, 2, 1], [3, 2, 1]])
    x = layer.forward(x)
    print(x.shape)