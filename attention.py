import maths
from maths import Linear

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

        values = self.values.forward(values)
        keys = self.keys.forward(keys)
        queries = self.queries.forward(queries)
