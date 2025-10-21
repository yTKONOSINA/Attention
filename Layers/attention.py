import maths
from Layers.linear import Linear
from tensor import Tensor
import json

class BertSelfAttention:
    def __init__(self, hidden_size, num_heads):
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Linear projections
        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)

        # Output projection
        self.dense = Linear(hidden_size, hidden_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]

        # Q = XW_q + b_q, K = XW_k + b_k, V = XW_v + b_v
        values = self.values.forward(values)
        keys = self.keys.forward(keys)
        queries = self.queries.forward(queries)

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Attention(Q, K, V) = softmax(QK^t/sqrt(d_k))V
        # Output = Attention(Q, K, W)W_o + b_o

        values = values.reshape((N, value_len, self.num_heads, self.head_dim))
        keys = keys.reshape((N, key_len, self.num_heads, self.head_dim))
        queries = queries.reshape((N, query_len, self.num_heads, self.head_dim))

        #queries = queries.permute((0, 2, 1, 3)).reshape(N * self.num_heads, query_len, self.head_dim)
        #keys = keys.permute(0, 2, 1, 3).reshape(N * self.heads, key_len, self.head_dim)

        # I must incorporate mask, otherwise the model would consider it
        # To be continued ...
        return
    
class BertLayer:
    def __init__(self):
        pass

    def _load_weight(self):
        pass