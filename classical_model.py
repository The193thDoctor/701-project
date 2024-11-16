# classical_model.py

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from data_loader import download_subset_data, create_data_loader, DEVICE

# Device configuration
device = DEVICE


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, use_embeddings=False):
        super(SentimentClassifier, self).__init__()
        self.use_embeddings = use_embeddings
        if not use_embeddings:
            # Use BERT model for embeddings
            self.bert = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
            self.drop = nn.Dropout(p=0.3)
            self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        else:
            # Model for precomputed embeddings
            self.fc = nn.Linear(128, n_classes)  # Assuming embeddings are size 128

    def forward(self, inputs):
        if not self.use_embeddings:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs.pooler_output
            output = self.drop(pooled_output)
            return self.out(output)
        else:
            embeddings = inputs['embedding']
            output = self.fc(embeddings)
            return output


def train_epoch(model, data_loader, loss_fn, optimizer, device, use_embeddings):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        labels = batch['label'].to(device)

        if use_embeddings:
            embeddings = batch['embedding'].to(device).float()
            inputs = {'embedding': embeddings}
        else:
            texts = batch['text']
            tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
            encoding = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
            inputs = {'input_ids': encoding['input_ids'].to(device),
                      'attention_mask': encoding['attention_mask'].to(device)}

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)


def eval_model(model, data_loader, loss_fn, device, use_embeddings):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            labels = batch['label'].to(device)

            if use_embeddings:
                embeddings = batch['embedding'].to(device).float()
                inputs = {'embedding': embeddings}
            else:
                texts = batch['text']
                tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
                encoding = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
                inputs = {'input_ids': encoding['input_ids'].to(device),
                          'attention_mask': encoding['attention_mask'].to(device)}

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)


if __name__ == "__main__":
    # Parameters
    batch_size = 8
    use_embeddings = False  # Set to True to use precomputed embeddings
    num_epochs = 3
    n_classes = 2

    # Load data
    train_df, test_df = download_subset_data()
    train_loader = create_data_loader(train_df, batch_size=batch_size, use_embeddings=use_embeddings)
    test_loader = create_data_loader(test_df, batch_size=batch_size, use_embeddings=use_embeddings)

    # Initialize model
    model = SentimentClassifier(n_classes=n_classes, use_embeddings=use_embeddings)
    model = model.to(device)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, use_embeddings)
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        val_acc, val_loss = eval_model(model, test_loader, loss_fn, device, use_embeddings)
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), 'classical_model.bin')
