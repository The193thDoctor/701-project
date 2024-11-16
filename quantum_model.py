# quantum_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import download_subset_data, create_data_loader, DEVICE
import pennylane as qml

# Device configuration
device = DEVICE


class HybridQuantumClassifier(nn.Module):
    def __init__(self, n_qubits, n_layers, n_classes):
        super(HybridQuantumClassifier, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes

        # Quantum circuit parameters
        self.q_params = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))

        # Final classical layer
        self.fc = nn.Linear(n_qubits, n_classes)

    def quantum_layer(self, x):
        dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            for idx in range(self.n_qubits):
                qml.RY(inputs[idx], wires=idx)
            qml.templates.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        outputs = circuit(x, self.q_params)
        return outputs

    def forward(self, x):
        # x has shape (batch_size, n_qubits)
        quantum_outputs = torch.stack([self.quantum_layer(sample) for sample in x])
        logits = self.fc(quantum_outputs)
        return logits


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        embeddings = batch['embedding'].to(device).float()
        labels = batch['label'].to(device)

        # Reduce embeddings to match n_qubits
        embeddings = embeddings[:, :model.n_qubits]

        outputs = model(embeddings)
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

            # Reduce embeddings to match n_qubits
            embeddings = embeddings[:, :model.n_qubits]

            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(losses)


if __name__ == "__main__":
    # Parameters
    batch_size = 8
    use_embeddings = True  # Must be True for quantum model
    num_epochs = 3
    n_classes = 2
    n_qubits = 4
    n_layers = 2

    # Load data
    train_df, test_df = download_subset_data()
    train_loader = create_data_loader(train_df, batch_size=batch_size, use_embeddings=use_embeddings)
    test_loader = create_data_loader(test_df, batch_size=batch_size, use_embeddings=use_embeddings)

    # Initialize model
    model = HybridQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers, n_classes=n_classes)
    model = model.to(device)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        val_acc, val_loss = eval_model(model, test_loader, loss_fn, device)
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), 'quantum_model.bin')
