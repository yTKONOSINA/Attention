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

    def forward(self, X, mask):
        N, length, _ = X.shape

        # Q = XW_q + b_q, K = XW_k + b_k, V = XW_v + b_v
        values = self.value.forward(X)
        keys = self.key.forward(X)
        queries = self.query.forward(X)

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
        # Scale by sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = scores * scale
        if mask is not None:
            mask_tensor = Tensor(mask)
            expanded_mask = [[[[1 if (mask_tensor.tensor[b][i] == 0 or mask_tensor.tensor[b][j] == 0) else 0
                                for j in range(length)]
                               for i in range(length)]
                              for _ in range(self.num_heads)]
                             for b in range(N)]
            scores = scores.masked_fill(Tensor(expanded_mask), float("-inf"))
        attention = scores.softmax(dim=-1)

        output = attention @ values
        output = output.permute((0, 2, 1, 3)).reshape((N, length, self.hidden_size))

        output = self.dense.forward(output)
        return output
    
class BertLayer:
    def __init__(self, 
                 hidden_size = 128, 
                 intermediate_size = 512, 
                 num_heads = 2,
                 layer_num = 0,
                 weight_file = 'weights/encoder.json'):
        
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
        query_weight = Tensor(weights[prefix + "attention.self.query.weight"]).transpose_2d()
        self.attention.query.w = query_weight
        self.attention.query.b = Tensor(weights[prefix + "attention.self.query.bias"])
        
        key_weight = Tensor(weights[prefix + "attention.self.key.weight"]).transpose_2d()
        self.attention.key.w = key_weight
        self.attention.key.b = Tensor(weights[prefix + "attention.self.key.bias"])
        
        value_weight = Tensor(weights[prefix + "attention.self.value.weight"]).transpose_2d()
        self.attention.value.w = value_weight
        self.attention.value.b = Tensor(weights[prefix + "attention.self.value.bias"])
        
        dense_weight = Tensor(weights[prefix + "attention.output.dense.weight"]).transpose_2d()
        self.attention.dense.w = dense_weight
        self.attention.dense.b = Tensor(weights[prefix + "attention.output.dense.bias"])

        self.attention_norm.w = Tensor(weights[prefix + "attention.output.LayerNorm.weight"])
        self.attention_norm.b = Tensor(weights[prefix + "attention.output.LayerNorm.bias"])

        # Feed-forward
        intermediate_weight = Tensor(weights[prefix + "intermediate.dense.weight"]).transpose_2d()
        self.intermediate.w = intermediate_weight
        self.intermediate.b = Tensor(weights[prefix + "intermediate.dense.bias"])

        output_dense_weight = Tensor(weights[prefix + "output.dense.weight"]).transpose_2d()
        self.output_dense.w = output_dense_weight
        self.output_dense.b = Tensor(weights[prefix + "output.dense.bias"])

        self.output_norm.w = Tensor(weights[prefix + "output.LayerNorm.weight"])
        self.output_norm.b = Tensor(weights[prefix + "output.LayerNorm.bias"])
    

    def forward(self, hidden_states, mask=None):
        # Self-Attention + residual
        attn_out = self.attention.forward(hidden_states, mask)
        hidden_states = self.attention_norm.forward(hidden_states + attn_out)

        # Feed-Forward + residual
        intermediate_out = self.intermediate.forward(hidden_states)
        intermediate_out = self.gelu(intermediate_out)
        ff_out = self.output_dense.forward(intermediate_out)
        hidden_states = self.output_norm.forward(hidden_states + ff_out)

        return hidden_states

    def gelu(self, tensor: Tensor) -> Tensor:

        def gelu_fn(x):
            return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

        def apply_gelu(data):
            if not isinstance(data, list):
                return gelu_fn(data)
            return [apply_gelu(item) for item in data]
        
        return Tensor(apply_gelu(tensor.tensor))