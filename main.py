from transformers import AutoTokenizer # Only tokenizer
from embedding import embed
from attention import Attention


sentences = [
    "How are [MASK]?",
    "It is a [MASK] day.",
    "He is a good [MASK]."
]

# Loading the model and the tokenizer
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForMaskedLM.from_pretrained(model_name)

tokens_list = []
token_type_list = []
# word_embeddings: Tensor,
#           pos_embeddings: Tensor,
#           type_embeddings : Tensor,
#           layernorm_weight: Tensor,
#           layernorm_bias: Tensor,

# max token length is 512
# Tokenize with padding to the longest sentence
inputs = tokenizer(
    sentences,
    return_tensors="pt",
    padding=True,        # pads to longest sentence in batch
    truncation=True      # truncate if too long (optional)
)

tokens_list = inputs['input_ids'].tolist()
token_type_ids = input['token_type_ids'].tolist()

# Since we are working with batches, we have to pad each sentence to the longest
# one and create an output mask, otherwise the attention layer would consider
# these padded elements
mask = input['attention_mask'].tolist() 


# Create a separate file, which imports the model and 
# Get embeddings

# Get the results 

