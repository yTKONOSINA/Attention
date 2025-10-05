import maths

class SelfAttention:
    def __init__(self, embed_size, num_heads):
        self.embed_size = embed_size
        self.num_heads = num_heads

        assert (self.embed_size % self.num_heads == 0), \
        "The embedding size should be divisible by the number of heads"

        self.head_dim = embed_size // num_heads
