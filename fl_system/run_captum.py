import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
from model import ImprovedTransformer
from dataset_loader import load_and_preprocess_dataset
import numpy as np
import os

def run_captum_analysis():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    os.makedirs('explainability/captum', exist_ok=True)
    
    print("Loading datasets for Captum (robust sample size)...")
    train_loader, test_loader = load_and_preprocess_dataset("CICIoT", data_dir='../data', partition_id=None)
    
    test_data = []
    for i, (X, y) in enumerate(test_loader):
        if i == 0:
            test_data.append(X[:256]) # 256 samples
            break
            
    test_samples = torch.cat(test_data).cpu()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_samples = test_samples.to(device)
    test_samples.requires_grad_()
    
    feature_names = [f"F_{i}" for i in range(46)]
    
    models = {
        'Centralized': 'saved_models/centralized_model.pth',
        'Federated': 'saved_models/global_model.pth'
    }
    
    all_attributions = {}
    
    for model_name, model_path in models.items():
        print(f"Running Captum IntegratedGradients for {model_name} model...")
        model = ImprovedTransformer(input_dim=46, num_classes=34).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        ig = IntegratedGradients(model)
        
        preds = model(test_samples).argmax(dim=1)
        
        attributions, delta = ig.attribute(test_samples, target=preds, return_convergence_delta=True)
        
        attr_np = np.abs(attributions.cpu().detach().numpy()).mean(axis=0)
        all_attributions[model_name] = attr_np
        
        indices = np.argsort(attr_np)[-15:]
        plt.figure(figsize=(10, 6))
        plt.barh(np.arange(len(indices)), attr_np[indices], color='mediumpurple', edgecolor='black')
        plt.yticks(np.arange(len(indices)), [feature_names[i] for i in indices], fontweight='bold')
        plt.xlabel('Mean Absolute Integrated Gradients', weight='bold')
        plt.title(f'Top 15 Feature Importance - {model_name} (Captum)', weight='bold')
        plt.tight_layout()
        plt.savefig(f'explainability/captum/{model_name.lower()}_feature_importance.png', bbox_inches='tight', dpi=300)
        plt.close()

    print("Generating Captum Feature Importance Comparison Plot...")
    cent_attr = all_attributions['Centralized']
    fed_attr = all_attributions['Federated']
    
    indices = np.argsort(cent_attr)[-15:]
    
    plt.figure(figsize=(10, 8))
    width = 0.35
    y = np.arange(len(indices))
    
    plt.barh(y - width/2, cent_attr[indices], width, label='Centralized', color='skyblue', edgecolor='black')
    plt.barh(y + width/2, fed_attr[indices], width, label='Federated', color='salmon', edgecolor='black')
    
    plt.yticks(y, [feature_names[i] for i in indices], fontweight='bold')
    plt.xlabel('Mean Absolute Integrated Gradients', weight='bold')
    plt.title('Top 15 Feature Importance Comparison (Captum)', weight='bold')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('explainability/captum/comparison_feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Captum explainability completed.")

if __name__ == "__main__":
    run_captum_analysis()
