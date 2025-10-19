from transformers import AutoTokenizer
from embedding import embed
from attention import Attention


sentences = [
#    "the man went to the [MASK] . he bought a gallon of milk.",
#    "i wrote this passage with a [MASK]",
    "how are [MASK]?"
]

# Loading the model and the tokenizer
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForMaskedLM.from_pretrained(model_name)

# max token length is 512
for sentence in sentences:
    # Tokenizer
    # Return a batch (# sentences, # different lengths)
    tokens = tokenizer(sentence, return_tensors = "pt")


    print(tokens)
    token_type_ids = [[0] * len(sentence)]

# Get embeddings
# 1. First, get the weights from the model (the only part where we have to 
# use the tranformer library)
# Get 
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
embedding_weights = model.embeddings.word_embeddings.weight.detach().cpu().numpy().tolist()  # shape (30522,128)

# Get the results 