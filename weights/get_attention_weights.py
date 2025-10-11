from transformers import AutoModel, AutoConfig
import torch, json

model_name = "prajjwal1/bert-tiny"
model = AutoModel.from_pretrained(model_name, return_dict=False)
state = model.state_dict()

attention_weights = {}
for k, v in state.items():
    if "token_type_embeddings" in k:
        print(k, v)
    attention_weights[k] = v.detach().cpu().numpy().tolist()