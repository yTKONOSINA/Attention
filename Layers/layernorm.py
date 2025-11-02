from tensor import Tensor
import random

class LayerNorm:
    def __init__(self,
                 n : int,
                 w : list = None,
                 b : list = None):
        self.m = n
        if w:
            self.w = Tensor(w)
        else:
            self.w = Tensor(random.random() for _ in range(n))
        if b:
            self.b = Tensor(b)
        else:
            self.b = Tensor([random.random() for _ in range(n)])

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
