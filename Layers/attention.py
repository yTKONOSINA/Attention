import maths
import math
from Layers.linear import Linear
from tensor import Tensor
import json
from Layers.layernorm import LayerNorm

class BertSelfAttention:
    def __init__(self,
                 hidden_size,
                 num_heads
                 ):
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
        self.norm = LayerNorm(hidden_size)

    def forward(self, X, mask):
        N, length, _ = X.shape

        # Q = XW_q + b_q, K = XW_k + b_k, V = XW_v + b_v
        values = self.values.forward(X)
        keys = self.keys.forward(X)
        queries = self.queries.forward(X)

        # Attention(Q, K, V) = softmax(QK^t/sqrt(d_k))V
        # Output = Attention(Q, K, W)W_o + b_o

        values = values.reshape((N, length, self.num_heads, self.head_dim))
        keys = keys.reshape((N, length, self.num_heads, self.head_dim))
        queries = queries.reshape((N, length, self.num_heads, self.head_dim))

        queries = queries.permute((0, 2, 1, 3))
        keys = keys.permute((0, 2, 1, 3))
        values = values.permute((0, 2, 1, 3))
        
        KT = keys.transpose(-2, -1)
        scores = (queries @ KT)
        scores = Tensor([[[(val / math.sqrt(self.head_dim)) for val in row] for row in batch]
                         for batch in scores.tensor])
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attention = scores.softmax(dim=-1)

        output = attention @ values
        output = output.permute((0, 2, 1, 3)).reshape(N, length, self.hidden_size)

        output = self.dense.forward(output)
    
        output = self.norm.forward(output)
        
        return output
    
class BertLayer:
    def __init__(self, 
                 hidden_size = 128, 
                 intermediate_size = 512, 
                 num_heads = 2,
                 layer_num = 0,
                 weight_file = 'weight/encoder.json'):
        
        self._load_weights(weight_file, layer_num)

        self.attention = BertSelfAttention(hidden_size, num_heads)
        self.attention_norm = LayerNorm(hidden_size)

        self.intermediate = Linear(hidden_size, intermediate_size)
        self.output_dense = Linear(intermediate_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)

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
    

    def forward(self, hidden_states, mask=None):
        # Self-Attention + residual
        attn_out = self.attention.forward(hidden_states, mask)
        hidden_states = self.attention_norm.forward(hidden_states + attn_out)

        # Feed-Forward + residual
        ff_out = self.output_dense.forward(self.intermediate.forward(hidden_states))
        hidden_states = self.output_norm.forward(hidden_states + ff_out)

        return hidden_states