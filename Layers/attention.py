import maths
from Layers.linear import Linear
from tensor import Tensor
import json
from layernorm import LayerNorm

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
    def __init__(self, 
                 hidden_size = 128, 
                 intermediate_size = 512, 
                 num_heads = 2,
                 layer_num = 0,
                 weight_file = 'weight/encoder.json'):
        
        self.attention = BertSelfAttention(hidden_size, num_heads)
        self.attention_norm = LayerNorm(hidden_size)

        self.intermediate = Linear(hidden_size, intermediate_size)
        self.output_dense = Linear(intermediate_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)
        
        self._load_weights(weight_file, layer_num)

    def _load_weights(self, weight_file, layer_num):
        with open(weight_file, "r") as f:
            weights = json.load(f)

        prefix = f"bert.encoder.layer.{layer_num}."

        # Attention
        self.attention.query.weight = Tensor(weights[prefix + "attention.self.query.weight"])
        self.attention.query.bias = Tensor(weights[prefix + "attention.self.query.bias"])
        self.attention.key.weight = Tensor(weights[prefix + "attention.self.key.weight"])
        self.attention.key.bias = Tensor(weights[prefix + "attention.self.key.bias"])
        self.attention.value.weight = Tensor(weights[prefix + "attention.self.value.weight"])
        self.attention.value.bias = Tensor(weights[prefix + "attention.self.value.bias"])
        self.attention.dense.weight = Tensor(weights[prefix + "attention.output.dense.weight"])
        self.attention.dense.bias = Tensor(weights[prefix + "attention.output.dense.bias"])

        self.attention_norm.weight = Tensor(weights[prefix + "attention.output.LayerNorm.weight"])
        self.attention_norm.bias = Tensor(weights[prefix + "attention.output.LayerNorm.bias"])

        # Feed-forward
        self.intermediate.weight = Tensor(weights[prefix + "intermediate.dense.weight"])
        self.intermediate.bias = Tensor(weights[prefix + "intermediate.dense.bias"])

        self.output_dense.weight = Tensor(weights[prefix + "output.dense.weight"])
        self.output_dense.bias = Tensor(weights[prefix + "output.dense.bias"])

        self.output_norm.weight = Tensor(weights[prefix + "output.LayerNorm.weight"])
        self.output_norm.bias = Tensor(weights[prefix + "output.LayerNorm.bias"])