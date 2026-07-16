import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import ImprovedTransformer
from dataset_loader import load_and_preprocess_dataset
import os
import json
import random

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

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return acc, prec, rec, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading CICIoT test dataset for evaluation...")
    _, testloader = load_and_preprocess_dataset("CICIoT", data_dir='../data', partition_id=None)
    
    # Initialize Models
    centralized_model = ImprovedTransformer(input_dim=46, num_classes=34).to(device)
    fl_model = ImprovedTransformer(input_dim=46, num_classes=34).to(device)
    
    # Load Weights
    centralized_path = "saved_models/centralized_model.pth"
    fl_path = "saved_models/global_model.pth"
    
    c_ready = os.path.exists(centralized_path)
    f_ready = os.path.exists(fl_path)
    
    if c_ready:
        centralized_model.load_state_dict(torch.load(centralized_path, map_location=device))
    else:
        print(f"Warning: {centralized_path} not found.")
        
    if f_ready:
        fl_model.load_state_dict(torch.load(fl_path, map_location=device))
    else:
        print(f"Warning: {fl_path} not found.")
        
    if not (c_ready or f_ready):
        print("Models not found. Run training scripts first.")
        return
        
    print("Evaluating Centralized Model...")
    c_acc, c_prec, c_rec, c_f1 = evaluate_model(centralized_model, testloader, device) if c_ready else (0,0,0,0)
    
    print("Evaluating Federated Model...")
    f_acc, f_prec, f_rec, f_f1 = evaluate_model(fl_model, testloader, device) if f_ready else (0,0,0,0)
    
    # Print Results in Table Format
    results = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Centralized": [f"{c_acc:.4f}", f"{c_prec:.4f}", f"{c_rec:.4f}", f"{c_f1:.4f}"],
        "Federated": [f"{f_acc:.4f}", f"{f_prec:.4f}", f"{f_rec:.4f}", f"{f_f1:.4f}"]
    }
    
    df = pd.DataFrame(results)
    print("\n=== Evaluation Results ===")
    print(df.to_string(index=False))
    
    with open("results.json", "w") as f:
        json.dump(results, f)
    print("Saved evaluation results to results.json")

if __name__ == "__main__":
    main()
