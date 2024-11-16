# data_loader.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# Device configuration
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"  # BERT model for embeddings

def download_subset_data(train_size=2000, test_size=500, seed=10701):
    '''
    Downloads the Yelp Polarity dataset from Hugging Face Datasets using Pandas.
    Subsets both the train and test sets.
    Args:
        train_size (int): Number of rows to sample for the training set.
        test_size (int): Number of rows to sample for the test set.
        seed (int): Random seed for reproducibility.
    Returns:
        train_df (pd.DataFrame): Training data with 'text' and 'label' columns.
        test_df (pd.DataFrame): Testing data with 'text' and 'label' columns.
    '''
    splits = {
        'train': 'plain_text/train-00000-of-00001.parquet',
        'test': 'plain_text/test-00000-of-00001.parquet'
    }
    train_df = pd.read_parquet("hf://datasets/fancyzhx/yelp_polarity/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/fancyzhx/yelp_polarity/" + splits["test"])

    # Subset the data
    train_df_subset = train_df.sample(train_size, random_state=seed)
    test_df_subset = test_df.sample(test_size, random_state=seed)

    return train_df_subset, test_df_subset

def get_embeddings(list_of_texts, device=DEVICE):
    '''
    Get BERT embeddings for a given list of texts.
    Args:
        list_of_texts (list): List of strings to process.
        device (str): Device for computation ('cpu' or 'cuda').
    Returns:
        np.array: Embeddings as numpy arrays.
    '''
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).to(device)

    inputs = tokenizer(list_of_texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    outputs = model(**inputs)
    avg_hidden_states = torch.mean(outputs.last_hidden_state, dim=1)
    return avg_hidden_states.cpu().detach().numpy()

class YelpDataset(Dataset):
    '''
    Custom Dataset class for Yelp Polarity data.
    '''
    def __init__(self, texts, labels, embeddings=None):
        self.texts = texts
        self.labels = labels
        self.embeddings = embeddings  # Optionally pre-computed embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            return {'embedding': self.embeddings[idx], 'label': self.labels[idx]}
        else:
            return {'text': self.texts[idx], 'label': self.labels[idx]}

def create_data_loader(df, batch_size=16, use_embeddings=False, device=DEVICE):
    '''
    Creates a DataLoader for the given DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'label' columns.
        batch_size (int): Batch size for DataLoader.
        use_embeddings (bool): If True, precompute embeddings.
        device (str): Device for computation ('cpu' or 'cuda').
    Returns:
        DataLoader: PyTorch DataLoader.
    '''
    if use_embeddings:
        embeddings = get_embeddings(df['text'].tolist(), device)
        dataset = YelpDataset(texts=None, labels=df['label'].tolist(), embeddings=embeddings)
    else:
        dataset = YelpDataset(texts=df['text'].tolist(), labels=df['label'].tolist())

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Example usage of the data loader
    train_df, test_df = download_subset_data()

    # Create data loaders
    train_loader = create_data_loader(train_df, batch_size=8, use_embeddings=False)
    test_loader = create_data_loader(test_df, batch_size=8, use_embeddings=False)

    # Verify DataLoader output
    for batch in train_loader:
        print(batch['text'] if 'text' in batch else batch['embedding'])
        print(batch['label'])
        break