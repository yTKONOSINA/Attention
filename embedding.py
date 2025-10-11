from copy import deepcopy
from tensor import Tensor
from Layers.layernorm import LayerNorm

# embeddings.word_embeddings.weight torch.Size([30522, 128])
# embeddings.position_embeddings.weight torch.Size([512, 128])
# embeddings.token_type_embeddings.weight torch.Size([2, 128])
# embeddings.LayerNorm.weight torch.Size([128])
# embeddings.LayerNorm.bias torch.Size([128])

def embed(tokens : list[list[int]],
          token_type : list[list[int]],
          word_embeddings: Tensor,
          pos_embeddings: Tensor,
          type_embeddings : Tensor,
          layernorm_weight: Tensor,
          layernorm_bias: Tensor,
          ) -> Tensor:

    batch_embs = []

    max_len = max(len(sample) for sample in tokens)

    for ids, types in zip(tokens, token_type):
        seq_len = len(ids)
        
        assert len(ids) == len(types), "the length must match"
        
        # Padding each sample in the batch
        if seq_len < max_len:
            pad_len = max_len - seq_len
            ids = ids + [0] * pad_len
            types = types + [0] * pad_len

        # Word embeddings
        word_emb = Tensor([list(word_embeddings.tensor[i]) for i in ids])  # (seq_len, hidden)

        # Positional embeddings
        pos_emb = Tensor([list(pos_embeddings.tensor[i]) for i in range(max_len)])

        # Type embeddings
        type_emb = Tensor([list(type_embeddings.tensor[t]) for t in types])

        # Sum
        batch_embs.append((word_emb + pos_emb + type_emb).tensor)
    
    # batch layer norm

    x = Tensor(batch_embs)

    ln = LayerNorm(
                n = 128,
                w = layernorm_weight,
                b = layernorm_bias
                )
    
    x = ln.forward(x)

    return x