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
        
        res.tensor = [
            [row[i] + self.b.tensor[i] for i in range(self.n)]
            for row in res.tensor
        ]
        return res

if __name__ == "__main__":
    layer = Linear(3, 54, bias = 1, data = 1.0)
    x = Tensor([[3, 2, 1], [3, 2, 1]])
    x = layer.forward(x)
    print(x.shape)