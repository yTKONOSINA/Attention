from transformers import AutoTokenizer # Only tokenizer
from embedding import embed
from Layers.attention import BertLayer
from Layers.predictions import Predictions
from attention_visualization import visualize_attention
from tensor import Tensor
import json

sentences = [
    "Shakespeare wrote famous [MASK] like Hamlet.",
    "How are [MASK]?",
    "It is a [MASK] day.",
    "He is a good [MASK].",
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
                   Tensor(embeddings_dict['bert.embeddings.LayerNorm.bias']))

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

output, attn0 = bert_layer_0.forward(embeddings, mask)
output, attn1 = bert_layer_1.forward(output, mask)

VISUALIZE = True
if VISUALIZE:
    first_sentence_tokens = tokenizer.convert_ids_to_tokens(tokens_list[0])
    visualize_attention([attn0, attn1], first_sentence_tokens)

predictions = Predictions(hidden_size = 128)
logits = predictions.forward(output)

def top_k(scores, k):
    scores_list = list(enumerate(scores))
    scores_list.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in scores_list[:k]]

k = 5
logits_tensor = logits.tensor
mask_token_id = tokenizer.mask_token_id # 103

for batch_idx, token_ids in enumerate(tokens_list):
    mask_positions = [pos for pos, token_id in enumerate(token_ids) if token_id == mask_token_id]
    if not mask_positions:
        continue

    print(f"\nInput: {sentences[batch_idx]}")
    print("Top predictions for [MASK]:")

    for pos in mask_positions:
        vocab_scores = logits_tensor[batch_idx][pos]
        top_ids = top_k(vocab_scores, k)
        top_tokens = [tokenizer.decode([token_id]).strip() for token_id in top_ids]
        print(f"  Position {pos}: {', '.join(top_tokens)}")
