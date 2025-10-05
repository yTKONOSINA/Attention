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

def linear(m : int, n : int, a : float = 0) -> list:
    return [[a for i in range(n)] for j in range(m)]

if __name__ == "__main__":
    print(linear(3, 2))