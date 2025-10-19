from transformers import AutoModelForMaskedLM
from copy import deepcopy
from tensor import Tensor
from Layers.layernorm import LayerNorm

def embed(tokens : list[list[int]],
          token_type : list[list[int]],
          word_embeddings: Tensor,
          pos_embeddings: Tensor,
          type_embeddings : Tensor,
          layernorm_weight: Tensor,
          layernorm_bias: Tensor,
          ) -> Tensor:

    batch_embs = []
    for ids, types in zip(tokens, token_type):

        # Word embeddings
        word_emb = Tensor([list(word_embeddings.tensor[i]) for i in ids])  # (seq_len, hidden)

        # Positional embeddings
        pos_emb = Tensor([list(pos_embeddings.tensor[i]) for i in range(len(tokens[0]))])

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