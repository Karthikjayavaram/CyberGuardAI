import flwr as fl
import torch
import os
from collections import OrderedDict
from model import ImprovedTransformer
# pyrefly: ignore [missing-import]
from flwr.common import parameters_to_ndarrays
import random
import numpy as np

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

import json

SAVE_PATH = "saved_models/global_model.pth"
os.makedirs("saved_models", exist_ok=True)

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fl_round_accuracies = []

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_metrics is not None and "accuracy" in aggregated_metrics:
            self.fl_round_accuracies.append(aggregated_metrics["accuracy"])
            
        if server_round == 10:
            with open("fl_metrics.json", "w") as f:
                json.dump(self.fl_round_accuracies, f)
            print("Saved FL round accuracies to fl_metrics.json")
            
        return aggregated_loss, aggregated_metrics

    def aggregate_fit(self, server_round, results, failures):
        print(f"Server received parameters from {len(results)} clients", flush=True)
        print("Aggregation started", flush=True)
        # Call original FedAvg aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert aggregated parameters to numpy ndarrays
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            
            # Initialize the model (using CICIoT dimensions)
            model = ImprovedTransformer(input_dim=46, num_classes=34)
            
            # Update model with aggregated parameters
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            # Save the global model for this round
            round_path = f"saved_models/global_model_round_{server_round}.pth"
            torch.save(model.state_dict(), round_path)
            print(f"Saved global model after round {server_round} to {round_path}")
            
            # If this is the last round, save the final model
            if server_round == 10:
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"Final global model saved to {SAVE_PATH}")
        print("Aggregation finished", flush=True)
        print(f"Round completed (Round {server_round})", flush=True)
        return aggregated_parameters, aggregated_metrics

def main():
    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
