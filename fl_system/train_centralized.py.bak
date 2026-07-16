import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from model import ImprovedTransformer
from dataset_loader import load_and_preprocess_dataset
import os
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def main():
    print("Loading CICIoT dataset for centralized training...")
    trainloader, testloader = load_and_preprocess_dataset("CICIoT", data_dir='../data', partition_id=None)
    
    centralized_trainloader = trainloader
    centralized_testloader = testloader
    merged_train = trainloader.dataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedTransformer(input_dim=46, num_classes=34).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Centralized model is trained under comparable optimization settings to FL for fair empirical comparison, not exact compute equivalence.
    epochs = 10 # 10 passes over the entire dataset (matches FL: 10 rounds * 1 local epoch)
    
    print(f"Starting centralized training for {epochs} epochs on {len(merged_train)} samples...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in centralized_trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(centralized_trainloader):.4f}")
        
    os.makedirs("saved_models", exist_ok=True)
    save_path = "saved_models/centralized_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Centralized model saved to {save_path}")

if __name__ == "__main__":
    main()
