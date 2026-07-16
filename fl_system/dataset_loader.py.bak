import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

NUM_FEATURES = 46
NUM_CLASSES = 34
TOTAL_TRAIN_ROWS = 5491971
DEBUG_SAMPLE_SIZE = 50000  # Set to None for full-dataset mode

class IoTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def fit_and_save_preprocessors(train_path, encoder_path, scaler_path):
    print("Fitting LabelEncoder and StandardScaler on the full dataset using chunks...")
    le = LabelEncoder()
    scaler = StandardScaler()
    all_classes = set()
    
    for chunk in pd.read_csv(train_path, chunksize=500000):
        X_chunk = chunk.iloc[:, :-1].values
        y_chunk = chunk.iloc[:, -1].values
        scaler.partial_fit(X_chunk)
        all_classes.update(y_chunk)
        
    le.fit(list(all_classes))
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, encoder_path)
    print("Saved preprocessors successfully.")

def load_and_preprocess_dataset(dataset_name=None, data_dir='../data', partition_id=None, num_partitions=5):
    """
    Loads train.csv and test.csv from data_dir efficiently using partitions.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    encoder_path = os.path.join(data_dir, "label_encoder.joblib")
    scaler_path = os.path.join(data_dir, "scaler.joblib")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Warning: {train_path} or {test_path} not found. Generating dummy data.")
        X_train = np.random.randn(800, NUM_FEATURES)
        y_train = np.random.randint(0, NUM_CLASSES, size=800)
        X_test = np.random.randn(200, NUM_FEATURES)
        y_test = np.random.randint(0, NUM_CLASSES, size=200)
    else:
        # Precompute scaler and encoder if not exist
        if not os.path.exists(encoder_path) or not os.path.exists(scaler_path):
            fit_and_save_preprocessors(train_path, encoder_path, scaler_path)
            
        le = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        
        # Load train data (only assigned partition or iteratively sample all partitions)
        if partition_id is not None:
            partition_size = TOTAL_TRAIN_ROWS // num_partitions
            skip_rows = partition_id * partition_size
            num_rows = TOTAL_TRAIN_ROWS - skip_rows if partition_id == num_partitions - 1 else partition_size
            
            print(f"Loading Partition {partition_id}: {num_rows} samples (skipping {skip_rows})...")
            # skipping rows logic: range(1, skip_rows+1) ignores header but skips data lines. header remains on line 0
            skip_range = range(1, skip_rows + 1) if skip_rows > 0 else None
            df_train = pd.read_csv(train_path, skiprows=skip_range, nrows=num_rows)
            
            original_partition_size = len(df_train)
            
            # FL Client: Sample normally from its partition
            if DEBUG_SAMPLE_SIZE is not None and DEBUG_SAMPLE_SIZE < original_partition_size:
                try:
                    from sklearn.model_selection import train_test_split
                    _, df_train = train_test_split(df_train, test_size=DEBUG_SAMPLE_SIZE, stratify=df_train.iloc[:, -1], random_state=42)
                except ValueError:
                    df_train = df_train.sample(n=DEBUG_SAMPLE_SIZE, random_state=42)
                
            print(f"Original partition size:\n{original_partition_size:,}")
            print(f"Sampled size:\n{len(df_train):,}")
            if DEBUG_SAMPLE_SIZE is not None:
                print(f"Sampling ratio:\n{(len(df_train)/original_partition_size)*100:.2f}%\n")
                
        else:
            if DEBUG_SAMPLE_SIZE is not None:
                # Centralized Baseline: Reconstruct the exact same sampled data iteratively to save memory
                print(f"Iteratively loading and sampling {num_partitions} partitions for centralized baseline...")
                sampled_partitions = []
                original_full_size = 0
                partition_size = TOTAL_TRAIN_ROWS // num_partitions
                
                for p_id in range(num_partitions):
                    skip_rows = p_id * partition_size
                    num_rows = TOTAL_TRAIN_ROWS - skip_rows if p_id == num_partitions - 1 else partition_size
                    skip_range = range(1, skip_rows + 1) if skip_rows > 0 else None
                    
                    # Read only this partition from CSV (index resets naturally starting from 0)
                    df_part = pd.read_csv(train_path, skiprows=skip_range, nrows=num_rows)
                    original_full_size += len(df_part)
                    
                    if DEBUG_SAMPLE_SIZE < len(df_part):
                        try:
                            from sklearn.model_selection import train_test_split
                            _, df_part = train_test_split(df_part, test_size=DEBUG_SAMPLE_SIZE, stratify=df_part.iloc[:, -1], random_state=42)
                        except ValueError:
                            df_part = df_part.sample(n=DEBUG_SAMPLE_SIZE, random_state=42)
                    sampled_partitions.append(df_part)
                
                # Concatenate the 5 identical client subsets
                df_train = pd.concat(sampled_partitions).reset_index(drop=True)
                print(f"Original full dataset size:\n{original_full_size:,}")
                print(f"Sampled size (reconstructed from {num_partitions} clients):\n{len(df_train):,}")
                print(f"Sampling ratio:\n{(len(df_train)/original_full_size)*100:.2f}%\n")
            else:
                # Full dataset mode
                print("Loading full train dataset...")
                df_train = pd.read_csv(train_path)
        X_train = df_train.iloc[:, :-1].values
        y_train_str = df_train.iloc[:, -1].values
        
        y_train = le.transform(y_train_str)
        X_train = scaler.transform(X_train)
        
        # Load test data (full test data for everyone, could be chunked if memory gets tight)
        print("Loading test dataset...")
        df_test = pd.read_csv(test_path)
        X_test = df_test.iloc[:, :-1].values
        y_test_str = df_test.iloc[:, -1].values
        
        y_test = le.transform(y_test_str)
        X_test = scaler.transform(X_test)
        
    # Create DataLoaders
    train_dataset = IoTDataset(X_train, y_train)
    test_dataset = IoTDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    return train_loader, test_loader
