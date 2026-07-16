import os
from PIL import Image
import pandas as pd
from docx import Document

files_to_check = [
    'plots/fl_accuracy_vs_round.png',
    'plots/fl_loss_vs_round.png',
    'plots/fl_convergence_analysis.png',
    'plots/centralized_vs_fl_accuracy.png',
    'plots/centralized_vs_fl_precision.png',
    'plots/centralized_vs_fl_recall.png',
    'plots/centralized_vs_fl_f1-score.png',
    'explainability/shap/centralized_summary_plot.png',
    'explainability/shap/centralized_bar_plot.png',
    'explainability/shap/federated_summary_plot.png',
    'explainability/shap/federated_bar_plot.png',
    'explainability/shap/comparison_feature_importance.png',
    'explainability/captum/centralized_feature_importance.png',
    'explainability/captum/federated_feature_importance.png',
    'explainability/captum/comparison_feature_importance.png',
    'comparison/comparison.csv',
    'comparison/comparison.xlsx',
    'reports/project_report.docx',
    'reports/project_report.pdf'
]

print("Starting strict validation...")
print("-" * 50)

all_passed = True

for f in files_to_check:
    if not os.path.exists(f):
        print(f"[FAIL] MISSING: {f}")
        all_passed = False
        continue
        
    size = os.path.getsize(f)
    if size < 500:
        print(f"[FAIL] SUSPICIOUSLY SMALL: {f} ({size} bytes)")
        all_passed = False
        continue
        
    # Content Validation
    try:
        if f.endswith('.png'):
            with Image.open(f) as img:
                img.verify()
        elif f.endswith('.csv'):
            pd.read_csv(f)
        elif f.endswith('.xlsx'):
            pd.read_excel(f)
        elif f.endswith('.docx'):
            Document(f)
        elif f.endswith('.pdf'):
            with open(f, 'rb') as pdf:
                header = pdf.read(4)
                if header != b'%PDF':
                    raise ValueError("Invalid PDF header")
        
        print(f"[OK] {f} | Size: {size/1024:.1f} KB | Integrity Verified")
        
    except Exception as e:
        print(f"[FAIL] CORRUPTED: {f} | Error: {e}")
        all_passed = False

print("-" * 50)
if all_passed:
    print("SUCCESS: All files verified strictly. No placeholders or corruption detected.")
else:
    print("ERROR: One or more files failed validation.")
