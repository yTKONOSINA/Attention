from transformers import AutoModelForMaskedLM
import json
import os

model_name = "prajjwal1/bert-tiny"
model = AutoModelForMaskedLM.from_pretrained(model_name)
state = model.state_dict()

# Saving embedding weights
embedding_file = "weights/embeddings.json"
if not os.path.exists(embedding_file):
    embeddings = {}
    for k, v in state.items():
        if "embedding" in k.lower():
            embeddings[k] = v.tolist()
    with open(embedding_file, "w") as f:
        json.dump(embeddings, f)
    print(f"Saved embeddings to {embedding_file}")

# Saving encoder weights
encoder_file = "weights/encoder.json"
if not os.path.exists(encoder_file):
    encoder = {}
    for k, v in state.items():
        if "encoder" in k.lower():
            encoder[k] = v.tolist()
    with open(encoder_file, "w") as f:
        json.dump(encoder, f)
    print(f"Saved encoder weights to {encoder_file}")

# Save prediction weights
pred_file = "weights/predictions.json"
if not os.path.exists(pred_file):
    predictions = {}
    for k, v in state.items():
        if "predictions" in k.lower():
            predictions[k] = v.tolist()
    with open(pred_file, "w") as f:
        json.dump(predictions, f)
    print(f"Saved prediction weights to {pred_file}")
