# compare_plot.py

import matplotlib.pyplot as plt
import torch
import pandas as pd
from classical_model import SentimentClassifier
from quantum_model import HybridModel
from data_loader import create_data_loader
from transformers import BertTokenizer


def evaluate_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset)


if __name__ == "__main__":
    df = pd.read_csv('yelp_polarity.csv')
    df['label'] = df['label'].map({1: 0, 2: 1})

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128
    batch_size = 16

    data_loader = create_data_loader(df, tokenizer, max_len, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load classical model
    classical_model = SentimentClassifier(n_classes=2)
    classical_model.load_state_dict(torch.load('classical_model.bin'))
    classical_model = classical_model.to(device)

    # Load quantum model
    n_qubits = 4
    n_layers = 2
    quantum_model = HybridModel(n_qubits, n_layers)
    quantum_model.load_state_dict(torch.load('quantum_model.bin'))
    quantum_model = quantum_model.to(device)

    # Evaluate models
    classical_acc = evaluate_model(classical_model, data_loader, device)
    quantum_acc = evaluate_model(quantum_model, data_loader, device)

    # Plotting
    models = ['Classical', 'Quantum']
    accuracies = [classical_acc.item(), quantum_acc.item()]

    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim([0, 1])
    plt.show()
