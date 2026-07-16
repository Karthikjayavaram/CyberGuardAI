import torch
import sys
sys.path.append('fl_system')
from model import ImprovedTransformer
import shap
import numpy as np
import joblib

device = torch.device("cpu")
model = ImprovedTransformer(input_dim=46, num_classes=34)
model.load_state_dict(torch.load('fl_system/saved_models/centralized_model.pth', map_location=device))
model.eval()

X_bg = torch.randn(256, 46)
X_test = torch.randn(1, 46)

explainer = shap.DeepExplainer(model, X_bg)
shap_values = explainer.shap_values(X_test, check_additivity=False)

if isinstance(shap_values, list):
    print("List of len", len(shap_values), "each shape", shap_values[0].shape)
else:
    print("Array of shape", shap_values.shape)
