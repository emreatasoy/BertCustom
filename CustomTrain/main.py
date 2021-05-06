import torch
from AtisDataSet import *

# Bu data okumak icin
from torch.utils.data import DataLoader

#Bu fine-tune edeceğimiz modeli indiriyor.
from transformers import DistilBertForSequenceClassification, AdamW


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Burada bir de class sayısının verilmesi gerekiyor argüman olarak. Ör: num_labels=5
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

model.to(device)

#Bu modelin train moda girmesini sağlıyor.
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
print(train_texts)
optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()