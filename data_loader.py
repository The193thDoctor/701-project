# data_loader.py

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torchvision import transforms
import pandas as pd


class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = YelpDataset(
        texts=df['text'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size)


if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv('yelp_polarity.csv')  # Make sure to have your dataset in CSV format
    df['label'] = df['label'].map({1: 0, 2: 1})  # Assuming labels are 1 and 2

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128
    batch_size = 16

    data_loader = create_data_loader(df, tokenizer, max_len, batch_size)
