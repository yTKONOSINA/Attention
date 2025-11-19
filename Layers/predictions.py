import json
import os
import sys
import math

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
    
    def _load_weights(self, weights_file):
        pass

    def gelu(self, tensor: Tensor) -> Tensor:

        def gelu_fn(x):
            return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

        def apply_gelu(data):
            if not isinstance(data, list):
                return gelu_fn(data)
            return [apply_gelu(item) for item in data]
        
        return Tensor(apply_gelu(tensor.tensor))