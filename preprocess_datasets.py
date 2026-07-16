import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# Universal Label Mapping for 10 classes
LABEL_MAPPING = {
    "benign": 0, "normal": 0, 0: 0, "0": 0,
    "dos": 1, "denial of service": 1,
    "ddos": 2, "distributed denial of service": 2,
    "port scanning": 3, "information gathering": 3, "scanning": 3,
    "brute force": 4, "password cracking": 4,
    "injection": 5, "sqli": 5, "xss": 5,
    "mitm": 6, "man in the middle": 6,
    "malware": 7, "botnet": 7, "mirai": 7, "bashlite": 7,
    "ransomware": 8,
    # Any other string will map to 9 (Other)
}

def map_label(label_val):
    if isinstance(label_val, str):
        label_val = label_val.lower().strip()
        for key, mapped_val in LABEL_MAPPING.items():
            if isinstance(key, str) and key in label_val:
                return mapped_val
        return 9  # Other
    elif isinstance(label_val, (int, float)):
        val = int(label_val)
        if 0 <= val <= 9:
            return val
        return 9
    return 9

def preprocess_dataset(raw_csv_path, output_train_path, output_test_path, num_features=43):
    print(f"Processing {raw_csv_path}...")
    try:
        df = pd.read_csv(raw_csv_path)
    except FileNotFoundError:
        print(f"File {raw_csv_path} not found. Skipping.")
        return

    # Assume label is the last column
    X_raw = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]

    # Convert features to numeric, drop strings/IPs
    X_numeric = X_raw.select_dtypes(include=[np.number])
    if X_numeric.shape[1] < num_features:
        print(f"Warning: Dataset only has {X_numeric.shape[1]} numeric features. Padding with zeros.")
        for i in range(num_features - X_numeric.shape[1]):
            X_numeric[f'pad_feature_{i}'] = 0.0

    # Handle missing values
    X_numeric = X_numeric.fillna(0)

    # Map labels to 0-9
    y_mapped = y_raw.apply(map_label).astype(int)

    # Feature Selection down to EXACTLY `num_features`
    if X_numeric.shape[1] > num_features:
        print(f"Selecting top {num_features} features using ANOVA F-value...")
        # Add small noise to prevent zero variance errors during f_classif
        X_numeric += np.random.rand(*X_numeric.shape) * 1e-6
        selector = SelectKBest(score_func=f_classif, k=num_features)
        X_selected = selector.fit_transform(X_numeric, y_mapped)
    else:
        X_selected = X_numeric.values

    # Train/Test Split (80/20) with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_mapped, test_size=0.2, random_state=42
        )

    # Combine back to DataFrame and save
    df_train = pd.DataFrame(X_train)
    df_train['label'] = y_train.values
    df_train.to_csv(output_train_path, index=False)

    df_test = pd.DataFrame(X_test)
    df_test['label'] = y_test.values
    df_test.to_csv(output_test_path, index=False)

    print(f"Successfully processed and split {raw_csv_path}")

if __name__ == "__main__":
    datasets = ["TON-IoT", "CIC-IoT", "Edge-IIoT", "N-BaIoT", "HL-IoT"]
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    for ds in datasets:
        raw_path = os.path.join(data_dir, f"{ds}.csv")
        train_path = os.path.join(data_dir, f"{ds}_train.csv")
        test_path = os.path.join(data_dir, f"{ds}_test.csv")
        preprocess_dataset(raw_path, train_path, test_path)
    
    print("Preprocessing complete!")
