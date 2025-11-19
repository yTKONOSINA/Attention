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