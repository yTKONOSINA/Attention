from transformers import BertTokenizer, AutoModel
from embedding import embed
from attention import Attention


sentences = [
    "the man went to the [MASK] . he bought a gallon of milk.",
    "i wrote this passage with a [MASK]"
]
# max token length is 512
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
for sentence in sentences:
    tokens = tokenizer.tokenize(sentence,
                                    add_special_tokens=True) # [CLS], [SEP]
token_typs = [[0] * len(each sentence)]

model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
embedding_weights = model.embeddings.word_embeddings.weight.detach().cpu().numpy().tolist()  # shape (30522,128)