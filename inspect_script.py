import pandas as pd
import numpy as np

for file in ['data/train.csv', 'data/test.csv', 'data/validation.csv']:
    print(f"--- {file} ---")
    df = pd.read_csv(file)
    print("Rows:", len(df), "Cols:", len(df.columns))
    label_col = df.columns[-1]
    print("Label col name:", label_col)
    
    unique_classes = df[label_col].unique()
    print("Unique classes count:", len(unique_classes))
    print("Label dtype:", df[label_col].dtype)
    
    features = df.drop(columns=[label_col])
    numeric_features = features.select_dtypes(include=[np.number])
    all_numeric = (len(features.columns) == len(numeric_features.columns))
    print("All features numeric:", all_numeric)
