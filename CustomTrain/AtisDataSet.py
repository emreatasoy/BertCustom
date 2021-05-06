import os
import torch
import torch.utils.data
from pathlib import Path
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split

class AtisDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_atis_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []

    text_file = "seq.in"
    label_file = 'label'

    with open(os.path.join(split_dir, text_file)) as f:
        for x in f:
            texts.append(x.rstrip("\n"))
    f.close()


    with open(os.path.join(split_dir, label_file)) as f:
        for x in f:
            labels.append(x.rstrip("\n"))
    f.close()


    return texts, labels

train_texts, train_labels = read_atis_split('atis/train')
# print(train_texts)
print(len(set(train_labels)))

test_texts, test_labels = read_atis_split('atis/test')

# print(test_texts)
print(len(set(test_labels)))

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = AtisDataset(train_encodings, train_labels)


val_dataset = AtisDataset(val_encodings, val_labels)
test_dataset = AtisDataset(test_encodings, test_labels)