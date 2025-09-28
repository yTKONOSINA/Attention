import random
from maths import matmul, ReLU

class MLP:
    def __init__(self, dims : list):
        self.layers : list[tuple] = []
        for i in range(len(dims) - 1):
            weight = [
                [random.random() for i in range(dims[i + 1])] for j in range(dims[i])
            ]
            bias = [random.random() for i in range(dims[i + 1])]
            self.layers.append((weight, bias))
        
    def forward(self, x):
        for i, (W, b) in enumerate(self.layers):
            z = matmul(x, W)
            z = [[z_row[j] + b[j] for j in range(len(b))] for z_row in z]

            # Apply ReLU except for the last layer
            if i < len(self.layers) - 1:
                z = [ReLU(row) for row in z]

            x = z
        return x

if __name__ == "__main__":
    input = [[1.0, 2.0]]
    mlp = MLP([2, 3, 1])
    print(mlp.forward(input))