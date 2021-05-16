import json
import os
import pickle

import torch
import torch.utils.data
from pathlib import Path
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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


    with open(os.path.join(split_dir, label_file)) as labelFile:
        for x in labelFile:
            labels.append(x.rstrip("\n"))
    labelFile.close()


    return texts, labels

le = preprocessing.LabelEncoder()

train_texts, train_labels = read_atis_split('atis/train')
# print(train_texts)
train_labels_len = len(train_labels)
print("train label len : " + str(train_labels_len))
print(len(set(train_labels)))

test_texts, test_labels = read_atis_split('atis/test')
test_labels_len = len(test_labels)
print("test label len : " + str(test_labels_len))
print(len(set(test_labels)))

total_labels = []
total_labels.extend(train_labels)
total_labels.extend(test_labels)
print(len(total_labels))

total_labels_int = le.fit_transform(total_labels)
mapping = dict(zip(le.classes_, range(1, len(le.classes_)+1)))
print(mapping)

file_mapping = open(r"mapping.json", "wb")
with open('mapping.json', 'w') as file:
    file.write(json.dumps(mapping))
file_mapping.close()

total_different_labels = len(set(total_labels_int))
print("total different labels : " + str(total_different_labels))

train_labels_int = total_labels_int[:train_labels_len]
test_labels_int = total_labels_int[train_labels_len:]
print("train label len : " + str(len(train_labels_int)))
print("test label len : " + str(len(test_labels_int)))

train_texts, val_texts, train_labels_int, val_labels = train_test_split(train_texts, train_labels_int, test_size=.2)


print(len(set(train_labels_int)))
print(len(set(val_labels)))
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = AtisDataset(train_encodings, train_labels_int)
print(len(train_dataset))

item = {key: torch.tensor(val[3058]) for key, val in train_dataset.encodings.items()}
item['labels'] = torch.tensor(train_dataset.labels[3058])


val_dataset = AtisDataset(val_encodings, val_labels)
test_dataset = AtisDataset(test_encodings, test_labels_int)

print("burdaaaaaaaaaaaaaaaaaaaa")