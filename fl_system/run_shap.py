import torch
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import ImprovedTransformer
from dataset_loader import load_and_preprocess_dataset
import os

def run_shap_analysis():
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    os.makedirs('explainability/shap', exist_ok=True)
    
    print("Loading datasets for robust SHAP analysis...")
    train_loader, test_loader = load_and_preprocess_dataset("CICIoT", data_dir='../data', partition_id=None)
    
    background_data = []
    test_data = []
    
    # Load more data for a denser, more professional plot
    for i, (X, y) in enumerate(test_loader):
        if i == 0:
            background_data.append(X[:256]) # 256 for background
            test_data.append(X[256:512])    # 256 to explain
            break
            
    background = torch.cat(background_data).cpu()
    test_samples = torch.cat(test_data).cpu()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    background = background.to(device)
    test_samples = test_samples.to(device)
    
    feature_names = [f"F_{i}" for i in range(46)]
    
    models = {
        'Centralized': 'saved_models/centralized_model.pth',
        'Federated': 'saved_models/global_model.pth'
    }
    
    all_shap_values = {}
    
    for model_name, model_path in models.items():
        print(f"Running SHAP for {model_name} model (this may take a while)...")
        model = ImprovedTransformer(input_dim=46, num_classes=34).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_samples, check_additivity=False)
        
        # Proper multidimensional mean handling for PyTorch outputs
        if isinstance(shap_values, list):
            mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
            shap_values_np = np.mean(mean_abs_per_class, axis=0) # shape (features,)
        else:
            shap_values_np = np.abs(shap_values)
            if shap_values_np.ndim > 2:
                feature_axis = shap_values_np.shape.index(46)
                axes_to_mean = tuple(i for i in range(shap_values_np.ndim) if i != feature_axis)
                shap_values_np = np.mean(shap_values_np, axis=axes_to_mean)
            else:
                shap_values_np = np.mean(shap_values_np, axis=0)
                
        all_shap_values[model_name] = shap_values_np
            
        # Summary Plot (Beeswarm)
        plt.figure()
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], test_samples.cpu().numpy(), feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, test_samples.cpu().numpy(), feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary - {model_name}', weight='bold')
        plt.tight_layout()
        plt.savefig(f'explainability/shap/{model_name.lower()}_summary_plot.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Bar Plot
        plt.figure()
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], test_samples.cpu().numpy(), plot_type="bar", feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, test_samples.cpu().numpy(), plot_type="bar", feature_names=feature_names, show=False)
        plt.title(f'SHAP Feature Importance - {model_name}', weight='bold')
        plt.tight_layout()
        plt.savefig(f'explainability/shap/{model_name.lower()}_bar_plot.png', bbox_inches='tight', dpi=300)
        plt.close()

    print("Generating Feature Importance Comparison Plot...")
    cent_imp = all_shap_values['Centralized']
    fed_imp = all_shap_values['Federated']
    
    indices = np.argsort(cent_imp)[-15:] 
    
    plt.figure(figsize=(10, 8))
    width = 0.35
    y = np.arange(len(indices))
    
    plt.barh(y - width/2, cent_imp[indices], width, label='Centralized', color='skyblue', edgecolor='black')
    plt.barh(y + width/2, fed_imp[indices], width, label='Federated', color='salmon', edgecolor='black')
    
    plt.yticks(y, [feature_names[i] for i in indices], fontweight='bold')
    plt.xlabel('Mean Absolute SHAP Value', weight='bold')
    plt.title('Top 15 Feature Importance Comparison (SHAP)', weight='bold')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('explainability/shap/comparison_feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("SHAP explainability completed.")

if __name__ == "__main__":
    run_shap_analysis()
