import maths
from maths import Linear
from tensor import Tensor

class SelfAttention:
    def __init__(self, embed_size, num_heads):
        self.embed_size = embed_size
        self.num_heads = num_heads

        assert (self.embed_size % self.num_heads == 0), \
        "The embedding size should be divisible by the number of heads"

        self.head_dim = embed_size // num_heads

        self.values = Linear(self.embed_size, self.embed_size, bias = 0)
        self.keys = Linear(self.embed_size, self.embed_size, bias = 0)
        self.queries = Linear(self.embed_size, self.embed_size, bias = 0)
        # one more layer here for the output for cantatination of the heads

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]

        values = self.values.forward(values)
        keys = self.keys.forward(keys)
        queries = self.queries.forward(queries)

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape((N, value_len, self.num_heads, self.head_dim))
        keys = keys.reshape((N, key_len, self.num_heads, self.head_dim))
        queries = queries.reshape((N, query_len, self.num_heads, self.head_dim))

        #queries = queries.permute((0, 2, 1, 3)).reshape(N * self.num_heads, query_len, self.head_dim)
        #keys = keys.permute(0, 2, 1, 3).reshape(N * self.heads, key_len, self.head_dim)


        # To be continued ...
        return