from tensor import Tensor

class LayerNorm:
    def __init__(self, n : int, data : float = 0):
        self.m = n
        self.w = Tensor([data for i in range(n)])
        self.b = Tensor([data for i in range(n)])

    def forward(self, x : Tensor) -> Tensor:
        """
            LN(x) = (x - mean) / sqrt(std^2 + eps) * weight + bias
        """

        assert len(x.shape) == 3, "the input should be a batch"

        eps = 1e-12
        
        def norm_row(row):
            mean = sum(row)/len(row)
            var = sum((x - mean)**2 for x in row)/len(row)
            std = (var + eps) ** 0.5
            return [(x - mean) / std * w + b
                        for x, w, b in zip(row, self.w.tensor, self.b.tensor)]

        return Tensor([[norm_row(row) for row in sample] for sample in x.tensor])
