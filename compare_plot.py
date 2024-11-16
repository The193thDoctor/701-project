# compare_plot.py

import matplotlib.pyplot as plt
import torch
from data_loader import download_subset_data, create_data_loader, DEVICE
from classical_model import SentimentClassifier
from quantum_model import HybridQuantumClassifier
from transformers import BertTokenizer
import numpy as np

device = DEVICE


def evaluate_model(model, data_loader, device, use_embeddings, model_type='classical'):
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            labels = batch['label'].to(device)

            if use_embeddings:
                embeddings = batch['embedding'].to(device).float()
                if model_type == 'quantum':
                    embeddings = embeddings[:, :model.n_qubits]
                inputs = {'embedding': embeddings}
            else:
                texts = batch['text']
                tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
                encoding = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
                inputs = {'input_ids': encoding['input_ids'].to(device),
                          'attention_mask': encoding['attention_mask'].to(device)}

            outputs = model(inputs) if model_type == 'classical' else model(inputs['embedding'])
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset)


if __name__ == "__main__":
    # Parameters
    batch_size = 8
    use_embeddings = True  # Set to True to use embeddings
    n_classes = 2
    n_qubits = 4
    n_layers = 2

    # Load data
    train_df, test_df = download_subset_data()
    test_loader = create_data_loader(test_df, batch_size=batch_size, use_embeddings=use_embeddings)

    # Load classical model
    classical_model = SentimentClassifier(n_classes=n_classes, use_embeddings=use_embeddings)
    classical_model.load_state_dict(torch.load('classical_model.bin'))
    classical_model = classical_model.to(device)

    # Load quantum model
    quantum_model = HybridQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers, n_classes=n_classes)
    quantum_model.load_state_dict(torch.load('quantum_model.bin'))
    quantum_model = quantum_model.to(device)

    # Evaluate models
    classical_acc = evaluate_model(classical_model, test_loader, device, use_embeddings, model_type='classical')
    quantum_acc = evaluate_model(quantum_model, test_loader, device, use_embeddings, model_type='quantum')

    # Plotting
    models = ['Classical', 'Quantum']
    accuracies = [classical_acc.item(), quantum_acc.item()]

    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim([0, 1])
    plt.show()
