import torch
import shap
import matplotlib.pyplot as plt
import numpy as np
from model import ImprovedTransformer
from dataset_loader import load_and_preprocess_dataset

def generate_shap_comparison():
    print("Loading tiny dataset to regenerate missing SHAP comparison plot...")
    train_loader, test_loader = load_and_preprocess_dataset("CICIoT", data_dir='../data', partition_id=None)
    
    background_data = []
    test_data = []
    for i, (X, y) in enumerate(test_loader):
        if i == 0:
            background_data.append(X[:50])
            test_data.append(X[50:60])
            break
            
    background = torch.cat(background_data).cpu()
    test_samples = torch.cat(test_data).cpu()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    background = background.to(device)
    test_samples = test_samples.to(device)
    
    feature_names = [f"Feature_{i}" for i in range(46)]
    models = {
        'Centralized': 'saved_models/centralized_model.pth',
        'Federated': 'saved_models/global_model.pth'
    }
    
    all_shap_values = {}
    
    for model_name, model_path in models.items():
        print(f"Calculating SHAP for {model_name}...")
        model = ImprovedTransformer(input_dim=46, num_classes=34).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_samples, check_additivity=False)
        
        # Proper dimensionality reduction for PyTorch multiclass (list of arrays)
        if isinstance(shap_values, list):
            # shap_values is a list of length 'classes' (34). 
            # Each element is (batch, features)
            # We want to take absolute mean across batch, and then mean across classes
            mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
            shap_values_np = np.mean(mean_abs_per_class, axis=0) # shape (features,)
        else:
            shap_values_np = np.abs(shap_values)
            # if tensor is (batch, classes, features) or similar
            if shap_values_np.ndim > 2:
                # mean over all axes except the one with size 46 (features)
                feature_axis = shap_values_np.shape.index(46)
                axes_to_mean = tuple(i for i in range(shap_values_np.ndim) if i != feature_axis)
                shap_values_np = np.mean(shap_values_np, axis=axes_to_mean)
            else:
                shap_values_np = np.mean(shap_values_np, axis=0)
                
        all_shap_values[model_name] = shap_values_np
        
    print("Generating comparison plot...")
    cent_imp = all_shap_values['Centralized']
    fed_imp = all_shap_values['Federated']
    
    indices = np.argsort(cent_imp)[-15:] # Top 15
    
    plt.figure(figsize=(10, 8))
    width = 0.35
    y = np.arange(len(indices))
    
    plt.barh(y - width/2, cent_imp[indices], width, label='Centralized', color='skyblue')
    plt.barh(y + width/2, fed_imp[indices], width, label='Federated', color='salmon')
    
    plt.yticks(y, [feature_names[i] for i in indices])
    plt.xlabel('Mean Absolute SHAP Value')
    plt.title('Top 15 Feature Importance Comparison (SHAP)')
    plt.legend()
    plt.savefig('explainability/shap/comparison_feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Successfully regenerated explainability/shap/comparison_feature_importance.png")

if __name__ == "__main__":
    generate_shap_comparison()
