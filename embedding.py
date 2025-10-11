from transformers import BertTokenizer, AutoModel
from copy import deepcopy
from tensor import Tensor

# embeddings.word_embeddings.weight torch.Size([30522, 128])
# embeddings.position_embeddings.weight torch.Size([512, 128])
# embeddings.token_type_embeddings.weight torch.Size([2, 128])
# embeddings.LayerNorm.weight torch.Size([128])
# embeddings.LayerNorm.bias torch.Size([128])

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
embedding_weights = model.embeddings.word_embeddings.weight.detach().cpu().numpy().tolist()  # shape (30522,128)

def embed(sentence : str) -> Tensor:
    tokens = tokenizer.tokenize(sentence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    embeddings = Tensor([deepcopy(embedding_weights[i]) for i in ids])

    