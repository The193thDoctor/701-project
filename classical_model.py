# classical_model.py


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import download_subset_data, create_data_loader, DEVICE


# Device configuration
device = DEVICE

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, input_dim, hidden_dim):
        super(SentimentClassifier, self).__init__()
        # Assuming embeddings are of size 128
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, embeddings):
        temp = self.fc1(embeddings)
        temp1 = self.relu(temp)
        output = self.fc2(temp1)
        return output


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        embeddings = batch['embedding'].to(device).float()
        labels = batch['label'].to(device)
        
        outputs = model(embeddings[:, :input_dim])
        loss = loss_fn(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)




def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            embeddings = batch['embedding'].to(device).float()
            labels = batch['label'].to(device)

            outputs = model(embeddings[:, :input_dim])
            loss = loss_fn(outputs, labels)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)




if __name__ == "__main__":
    # Parameters
    batch_size = 8
    num_epochs = 20
    n_classes = 2
    hidden_dim = 64

    input_dim = 16

    # Load data
    train_df, test_df = download_subset_data()
    train_loader = create_data_loader(train_df, batch_size=batch_size, use_embeddings=True)
    test_loader = create_data_loader(test_df, batch_size=batch_size, use_embeddings=True)

    # Initialize model
    model = SentimentClassifier(n_classes = n_classes, input_dim = input_dim, hidden_dim = hidden_dim)
    model = model.to(device)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        val_acc, val_loss = eval_model(model, test_loader, loss_fn, device)
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), 'classical_model.bin')



