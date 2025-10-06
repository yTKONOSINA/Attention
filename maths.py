def matmul(A : list, B : list) -> list:
    """
        Args:
            Matrix A and B
        Return:
            A @ B
    """
    m, n = len(A), len(A[0])
    m1, n1 = len(B), len(B[0])
    assert n == m1, "cannot multiply matrices, check dimentions"

    res = [
        [0.0 for i in range(n1)] for j in range(m)
    ]
    for i in range(m):
        for j in range(n1):
            sum = res[i][j]
            for k in range(n):
                sum += A[i][k] * B[k][j]
            res[i][j] = sum
    return res

def ReLU(a : list):
    return [max(0, i) for i in a]

# Create a softmax function

class Linear:
    # modify it to include batches
    def __init__(self, m : int, n : int, bias : bool = True, 
                 a : float = 0):
        self.m = m
        self.n = n
        self.w = [[a for i in range(n)] for j in range(m)]
        self.b = [a for i in range(n)]
        self.bias = bias
    
    def forward(self, x : list) -> list:
        y = matmul(x, self.w)
        return [y[0][i]+ self.bias * self.b[i] for i in range(self.n)]

# Create a permutation function
# Create a reshaping function

if __name__ == "__main__":
    layer = Linear(3, 2, 1, 2)
    x = [[3, 2, 1]]
    print(layer.forward(x))
    print(layer.w)