# data_loader.py

import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader 
from sentence_transformers import SentenceTransformer # Device configuration 
from sklearn.decomposition import PCA

DEVICE = 'cpu' 
print(f"Using device: {DEVICE}")
MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"
pca_dim = 16

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
    ''' Get embeddings for a given list of texts using the new SentenceTransformer model. 
    Args: 
        list_of_texts (list): List of strings to process. 
        device (str): Device for computation ('cpu' or 'cuda'). 
    Returns: np.array: Embeddings as numpy arrays. 
    ''' 
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True).to(device) 
    model.max_seq_length = 128
    print("Encoding sentences...")
    embeddings = model.encode(list_of_texts, convert_to_tensor=True, device=device) 
    print("DONE\n")
    return embeddings.cpu().numpy()

class YelpDataset(Dataset):
    '''
    Custom Dataset class for Yelp Polarity data.
    '''
    def __init__(self, texts, labels,  pca_dim=128, embeddings=None):
        self.texts = texts
        self.labels = labels
        self.embeddings = embeddings # Optionally pre-computed embeddings 
        if embeddings is not None and pca_dim is not None: 
            self.pca = PCA(n_components=pca_dim) 
            self.embeddings = self.pca.fit_transform(self.embeddings)

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        if self.embeddings is not None:
            return {'embedding': (self.embeddings[idx]), 'label': self.labels[idx]}
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

def pca(data_loader):
    pca_model = PCA(n_components = pca_dim)
    out_data = pca_model.fit_transform(data_loader)
    return out_data

if __name__ == "__main__":
    # Example usage of the data loader
    train_df, test_df = download_subset_data()

    # Create data loaders
    train_loader = create_data_loader(train_df, batch_size=8, use_embeddings=True)
    test_loader = create_data_loader(test_df, batch_size=8, use_embeddings=True)

    # train_loader = pca(train_loader)
    # test_loader = pca(test_loader)

    # Verify DataLoader output
    for batch in train_loader:
        print(batch['text'] if 'text' in batch else batch['embedding'])
        print(batch['label'])
        break
    # sentences = [ "The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium." ] 
    # embeddings = get_embeddings(sentences, device=DEVICE)
    # print(embeddings)