from transformers import AutoTokenizer # Only tokenizer
from embedding import embed
from Layers.attention import BertLayer
from Layers.predictions import Predictions
from tensor import Tensor
import json

sentences = [
    "How are [MASK]?",
    "It is a [MASK] day.",
    "He is a good [MASK].",
    "Shakespeare wrote famous [MASK] like Hamlet.",
    "The [MASK] War ended in 1945.",
    "Python is a [MASK] language.",
    "Machine [MASK] learns from data."
]

# Loading the model and the tokenizer
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForMaskedLM.from_pretrained(model_name)

tokens_list = []
token_type_list = []

# max token length is 512
# Tokenize with padding to the longest sentence
inputs = tokenizer(
    sentences,
    return_tensors="pt",
    padding=True,
    truncation=True
)

tokens_list = inputs['input_ids'].tolist()
token_type_list = inputs['token_type_ids'].tolist()

# Since we are working with batches, we have to pad each sentence to the longest
# one and create an output mask, otherwise the attention layer would consider
# these padded elements
mask = inputs['attention_mask'].tolist() 

# Get embedding weights
with open("weights/embeddings.json", "r") as f:
    embeddings_dict = json.load(f)

# Get embeddings
embeddings = embed(tokens_list,
                   token_type_list,
                   Tensor(embeddings_dict["bert.embeddings.word_embeddings.weight"]),
                   Tensor(embeddings_dict['bert.embeddings.position_embeddings.weight']),
                   Tensor(embeddings_dict['bert.embeddings.token_type_embeddings.weight']),
                   Tensor(embeddings_dict['bert.embeddings.LayerNorm.weight']),
                   Tensor(embeddings_dict['bert.embeddings.LayerNorm.bias']) 
                )

print(embeddings.shape) # (batch, num of tokens, embedding dim = 128)

bert_layer_0 = BertLayer(hidden_size = 128,
                         intermediate_size = 512,
                         num_heads = 2,
                         layer_num = 0,
                         weight_file='weights/encoder.json')

bert_layer_1 = BertLayer(hidden_size = 128,
                         intermediate_size = 512,
                         num_heads = 2,
                         layer_num = 1,
                         weight_file='weights/encoder.json')
output = bert_layer_0.forward(embeddings, mask)
output = bert_layer_1.forward(output, mask)
print(f"Bert Layer 1 output shape: {output.shape}")

predictions = Predictions(hidden_size = 128)
output = predictions.forward(output)
print(f"Predictions output shape: {output.shape}")