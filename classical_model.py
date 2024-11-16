# classical_model.py

import torch
import torch.nn as nn
from transformers import BertModel
from data_loader import create_data_loader
import pandas as pd


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


if __name__ == "__main__":
    df = pd.read_csv('yelp_polarity.csv')
    df['label'] = df['label'].map({1: 0, 2: 1})

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128
    batch_size = 16

    data_loader = create_data_loader(df, tokenizer, max_len, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentClassifier(n_classes=2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss().to(device)

    epochs = 3
    for epoch in range(epochs):
        acc, loss = train_epoch(model, data_loader, loss_fn, optimizer, device)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}, Accuracy: {acc}')

    torch.save(model.state_dict(), 'classical_model.bin')
