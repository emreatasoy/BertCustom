import time

from AtisDataSet import *

# Bu data okumak icin
from torch.utils.data import DataLoader

# Bu fine-tune edeceğimiz modeli indiriyor.
from transformers import DistilBertForSequenceClassification, AdamW

cuda_available = torch.cuda.is_available()

torch.zeros(1).cuda()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Burada bir de class sayısının verilmesi gerekiyor argüman olarak. Ör: num_labels=5
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                            num_labels=total_different_labels)

model.to(device)

# Bu modelin train moda girmesini sağlıyor.
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

i = 1
for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        b_input_ids, b_input_mask, b_labels = batch
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

        print("işlem sürüyor : " + str(i))
        i += 1
model.eval()

logits = []
true_labels = []

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

print('Evaluating...')
start_time = time.time()
for batch_no, batch in enumerate(test_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    logits.append(outputs[1])
    true_labels.append(labels)

    if batch_no % 1000 == 0 and batch_no != 0:
        elapsed_time = time.time() - start_time
        print("Batch " + str(batch_no) + " of " + str(len(test_loader)) + ". Elapsed: " + time.strftime("%H:%M:%S",
                                                                                                        time.gmtime(
                                                                                                            elapsed_time)))

predicted_labels = torch.argmax(torch.cat(logits), dim=1).cpu().numpy()
true_test_labels = torch.cat(true_labels).cpu().numpy()

print(predicted_labels)

PATH: str = "atis_trained_model.pt"

torch.save(model, PATH)
