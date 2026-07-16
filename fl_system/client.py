import argparse
import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from model import ImprovedTransformer
from dataset_loader import load_and_preprocess_dataset
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

def train(model, trainloader, epochs, device):
    print("Training started", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        print("Epoch started", flush=True)
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Batch {i} / {len(trainloader)} - Current loss: {loss.item():.4f}", flush=True)
        print("Epoch completed", flush=True)

def test(model, testloader, device):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(testloader.dataset), correct / total

class IoTClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("Client connected", flush=True)
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1, device=self.device)
        print("Returning parameters", flush=True)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client_id", type=int, choices=[1, 2, 3, 4, 5], required=True, help="Client ID")
    args = parser.parse_args()

    dataset_name = "CICIoT"
    print(f"Client {args.client_id} loading dataset: {dataset_name}")

    trainloader, testloader = load_and_preprocess_dataset(
        dataset_name=dataset_name, 
        data_dir='../data',
        partition_id=args.client_id - 1,
        num_partitions=5
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedTransformer(input_dim=46, num_classes=34).to(device)

    client = IoTClient(model, trainloader, testloader, device)
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)

if __name__ == "__main__":
    main()
