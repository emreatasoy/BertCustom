import json

import torch
from transformers import DistilBertTokenizerFast

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path: str = './atis_trained_model.pt'

query_text = "I want to rent a car in boston"

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(query_text, return_tensors='pt')  # Bu tokenizer train yapılan tokenizer ile aynı olmalı
input_ids = encodings["input_ids"]

input_ids = input_ids.to(device)
# Load
model = torch.load(model_path)

model.to(device)
model.eval()
with torch.no_grad():
    outputs = model(input_ids)

logits = outputs.logits
max_index = torch.argmax(logits)
max_index += 1
with open('mapping.json') as json_file:
    labels_of_indexes = json.load(json_file)
keys = [k for k, v in labels_of_indexes.items() if v == max_index]

print("query text : " + str(query_text) + " and intention: " + str(keys))
