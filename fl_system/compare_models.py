import json
import pandas as pd
import os

def generate_comparison():
    with open('results.json', 'r') as f:
        results = json.load(f)
        
    metrics = results['Metric']
    centralized = [float(x) for x in results['Centralized']]
    federated = [float(x) for x in results['Federated']]
    
    df = pd.DataFrame({
        'Metric': metrics,
        'Centralized': centralized,
        'Federated': federated
    })
    
    df['Difference (FL - Cent)'] = df['Federated'] - df['Centralized']
    
    # Round to 4 decimal places for professional output
    df = df.round(4)
    
    os.makedirs('comparison', exist_ok=True)
    df.to_csv('comparison/comparison.csv', index=False)
    
    # Use context manager for ExcelWriter
    with pd.ExcelWriter('comparison/comparison.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Metrics')
        
    print("Statistical comparison saved to comparison/comparison.csv and comparison/comparison.xlsx")

if __name__ == "__main__":
    generate_comparison()
