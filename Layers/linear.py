from tensor import Tensor

class Linear:
    def __init__(self, m : int, n : int, data : float = 0):
        self.m = m
        self.n = n
        self.w = Tensor([[data for i in range(n)] for j in range(m)])
        self.b = Tensor([data for i in range(n)])
    
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