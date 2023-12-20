import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils import normalize_zero_2_one

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.long)
        self.labels = torch.tensor(labels.values, dtype=torch.float)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

def dataloader():
    df = pd.read_csv('unmod.txt', sep='\t')
    df.sample(frac=1, random_state=42).reset_index(drop=True)

    min_len, max_len = float('inf'), 0
    for i in df['sequence']:
        min_len = len(i) if len(i) < min_len else min_len
        max_len = len(i) if len(i) > max_len else max_len

    X = df['sequence']
    y = normalize_zero_2_one(df['RT'])
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.1, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    amino_acids = set(''.join(X_train))
    idx2acid = dict(enumerate(sorted(amino_acids), start=1))
    acid2idx = dict(map(reversed, idx2acid.items()))

    for seq in range(len(X_train)):
        sample = []
        for i in X_train[seq]:
            sample.append(acid2idx[i])
        sample.extend([0 for i in (range(max_len - len(sample)))])
        X_train[seq] = sample
        
    for seq in range(len(X_val)):
        sample = []
        for i in X_val[seq]:
            sample.append(acid2idx[i])
        sample.extend([0 for i in (range(max_len - len(sample)))])
        X_val[seq] = sample
        
    for seq in range(len(X_test)):
        sample = []
        for i in X_test[seq]:
            sample.append(acid2idx[i])
        sample.extend([0 for i in (range(max_len - len(sample)))])
        X_test[seq] = sample

    batch_size = int(len(X_train) / 4)

    train_data = torch.utils.data.DataLoader(
        CustomDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    val_data = torch.utils.data.DataLoader(
        CustomDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    test_data = torch.utils.data.DataLoader(
        CustomDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    return train_data, val_data, test_data