import json
import os
import sys
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tensor import Tensor
from Layers.linear import Linear
from Layers.layernorm import LayerNorm

class Predictions:
    def __init__(self,
                 hidden_size : int = 128,
                 vocab_size : int = 30522, 
                 weights_file : str = 'weights/predictions.json'):
            
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size

            self.dense = Linear(hidden_size, hidden_size)
            self.norm = LayerNorm(hidden_size)
            self.decoder = Linear(hidden_size, vocab_size)

            self.bias = None

            self._load_weights(weights_file)
    
    def _load_weights(self, weights_file : str):
        with open(weights_file, "r") as f:
            weights = json.load(f)

        dense = Tensor(weights["cls.predictions.transform.dense.weight"]).transpose_2d()
        self.dense.w = dense
        self.dense.b = Tensor(weights["cls.predictions.transform.dense.bias"])

        norm = Tensor(weights["cls.predictions.transform.LayerNorm.weight"])
        self.norm.w = norm
        self.norm.b = Tensor(weights["cls.predictions.transform.LayerNorm.bias"])

        decoder = Tensor(weights["cls.predictions.decoder.weight"]).transpose_2d()
        self.decoder.w = decoder
        self.decoder.b = Tensor(weights["cls.predictions.decoder.bias"])

        self.bias = Tensor(weights["cls.predictions.bias"])

    def gelu(self, tensor: Tensor) -> Tensor:

        def gelu_fn(x):
            return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

        def apply_gelu(data):
            if not isinstance(data, list):
                return gelu_fn(data)
            return [apply_gelu(item) for item in data]
        
        return Tensor(apply_gelu(tensor.tensor))

    def add_bias(self, x : Tensor) -> Tensor:
        return x + self.bias

    def forward(self, x : Tensor) -> Tensor:
        x = self.add_bias(x)
        x = self.dense.forward(x)
        x = self.gelu(x)
        x = self.norm.forward(x)
        x = self.decoder.forward(x)
        return x

if __name__ == "__main__":
    import random

    batch_size, seq_len, hidden_size = 2, 3, 128
    x = Tensor([[[random.random()
                for _ in range(hidden_size)]
                for _ in range(seq_len)]
                for _ in range(batch_size)])
    
    predictions = Predictions(hidden_size = hidden_size)
    output = predictions.forward(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output tensor: {output.tensor}")