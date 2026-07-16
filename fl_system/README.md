# Federated Learning IoT IDS Experiment

## Experimental Validity Statement
- **No Data Leakage**: There is zero data leakage across clients. Each client node securely and exclusively loads its assigned dataset split.
- **Data Exclusivity**: Each FL client has an exclusive dataset (e.g., Client 1 strictly uses TON-IoT, Client 2 strictly uses CIC-IoT, etc.). No data is shared globally or exchanged between nodes.
- **Preprocessing Consistency**: The exact same 43-feature preprocessing pipeline and StandardScaling is used across the entire system.
- **Strict Evaluation**: Evaluation uses explicitly isolated and unseen test splits (20%) reserved independently from the local training process.

## Fairness Statement
- **Centralized Baseline**: The `train_centralized.py` baseline is trained under comparable optimization settings (same optimizer, LR, and equivalent passes over the full dataset) to the FL architecture for fair empirical comparison, ensuring research validity.
