from transformers import AutoModel, AutoConfig
import torch, json

model_name = "prajjwal1/bert-tiny"
model = AutoModel.from_pretrained(model_name, return_dict=False)
state = model.state_dict()

attention_weights = {}
for k, v in state.items():
    #if "attention" in k:
    print(k, v.shape)
    attention_weights[k] = v.detach().cpu().numpy().tolist()

# encoder.layer.0.attention.self.query.weight torch.Size([128, 128])
# encoder.layer.0.attention.self.query.bias torch.Size([128])

# encoder.layer.0.attention.self.key.weight torch.Size([128, 128])
# encoder.layer.0.attention.self.key.bias torch.Size([128])

# encoder.layer.0.attention.self.value.weight torch.Size([128, 128])
# encoder.layer.0.attention.self.value.bias torch.Size([128])

# encoder.layer.0.attention.output.dense.weight torch.Size([128, 128])
# encoder.layer.0.attention.output.dense.bias torch.Size([128])

# encoder.layer.0.attention.output.LayerNorm.weight torch.Size([128])
# encoder.layer.0.attention.output.LayerNorm.bias torch.Size([128])

# encoder.layer.1.attention.self.query.weight torch.Size([128, 128])
# encoder.layer.1.attention.self.query.bias torch.Size([128])
# encoder.layer.1.attention.self.key.weight torch.Size([128, 128])
# encoder.layer.1.attention.self.key.bias torch.Size([128])
# encoder.layer.1.attention.self.value.weight torch.Size([128, 128])
# encoder.layer.1.attention.self.value.bias torch.Size([128])
# encoder.layer.1.attention.output.dense.weight torch.Size([128, 128])
# encoder.layer.1.attention.output.dense.bias torch.Size([128])
# encoder.layer.1.attention.output.LayerNorm.weight torch.Size([128])
# encoder.layer.1.attention.output.LayerNorm.bias torch.Size([128])