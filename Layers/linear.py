from tensor import Tensor

class Linear:
    # modify it to include batches
    def __init__(self, m : int, n : int, bias : bool = True, 
                 data : float = 0):
        self.m = m
        self.n = n
        self.w = Tensor([[data for i in range(n)] for j in range(m)])
        self.bias = bias
        if bias:
            self.b = Tensor([data for i in range(n)])
        else:
            self.b = None
    
    def forward(self, x : Tensor) -> Tensor:
        y = x @ self.w
        
        # Broadcast the bias to every batch
        if self.bias:
            y.tensor = [
                [y_row[i] + self.b.tensor[i] for i in range(self.n)]
                for y_row in y.tensor
            ]
        return y

if __name__ == "__main__":
    layer = Linear(3, 54, bias = 1, data = 1.0)
    x = Tensor([[3, 2, 1], [3, 2, 1]])
    x = layer.forward(x)
    print(x.shape)