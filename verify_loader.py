import sys
import os

# add fl_system to path to allow importing dataset_loader
sys.path.append(os.path.join(os.path.dirname(__file__), 'fl_system'))
from dataset_loader import load_and_preprocess_dataset
import joblib

print("Testing dataloader with partitioning...")
try:
    train_loader, test_loader = load_and_preprocess_dataset(data_dir='data', partition_id=0, num_partitions=5)
    print("DataLoader works!")
    for X, y in train_loader:
        print(f"X batch shape: {X.shape}, y batch shape: {y.shape}")
        break
except Exception as e:
    print(f"Error: {e}")
