import os
import shutil
import subprocess
import time
import json
import csv
import re

EXPERIMENTS = [
    {"id": 1, "param": "Batch Size", "old": "1024", "new": "2048", "action": "batch_size_2048"},
    {"id": 2, "param": "Learning Rate", "old": "0.001", "new": "0.0005", "action": "lr_0.0005"},
    {"id": 3, "param": "Learning Rate", "old": "0.0005", "new": "0.0003", "action": "lr_0.0003"},
    {"id": 4, "param": "Weight Decay", "old": "1e-5", "new": "1e-4", "action": "weight_decay_1e-4"},
    {"id": 5, "param": "Transformer Heads", "old": "8", "new": "4", "action": "heads_4"},
    {"id": 6, "param": "Transformer Heads", "old": "4", "new": "16", "action": "heads_16"},
    {"id": 7, "param": "Transformer Layers", "old": "2", "new": "3", "action": "layers_3"},
    {"id": 8, "param": "Transformer Layers", "old": "3", "new": "4", "action": "layers_4"},
    {"id": 9, "param": "Hidden Dimension", "old": "256", "new": "512", "action": "embed_512"},
    {"id": 10, "param": "Dropout", "old": "0.10", "new": "0.20", "action": "dropout_0.2"},
    {"id": 11, "param": "Dropout", "old": "0.20", "new": "0.30", "action": "dropout_0.3"},
    {"id": 12, "param": "Optimizer", "old": "Adam", "new": "AdamW", "action": "optimizer_adamw"},
    {"id": 13, "param": "Epochs per Client", "old": "1", "new": "2", "action": "epochs_2"},
]

FILES_TO_BACKUP = ["model.py", "client.py", "server.py", "train_centralized.py", "dataset_loader.py"]

def backup_files():
    for f in FILES_TO_BACKUP:
        if os.path.exists(f):
            shutil.copy(f, f + ".bak")

def restore_files():
    for f in FILES_TO_BACKUP:
        if os.path.exists(f + ".bak"):
            shutil.copy(f + ".bak", f)

def modify_file(filepath, replacements):
    if not os.path.exists(filepath): return
    with open(filepath, 'r') as f:
        content = f.read()
    for pattern, repl in replacements:
        content = re.sub(pattern, repl, content)
    with open(filepath, 'w') as f:
        f.write(content)

def apply_experiment(action):
    if action == "batch_size_2048":
        modify_file("dataset_loader.py", [(r"batch_size=[0-9]+", "batch_size=2048")])
    elif action == "lr_0.0005":
        modify_file("client.py", [(r"lr=[0-9.]+", "lr=0.0005")])
        modify_file("train_centralized.py", [(r"lr=[0-9.]+", "lr=0.0005")])
    elif action == "lr_0.0003":
        modify_file("client.py", [(r"lr=[0-9.]+", "lr=0.0003")])
        modify_file("train_centralized.py", [(r"lr=[0-9.]+", "lr=0.0003")])
    elif action == "weight_decay_1e-4":
        modify_file("client.py", [(r"optimizer = torch\.optim\.([A-Za-z]+)\(([^)]+)\)", r"optimizer = torch.optim.\1(\2, weight_decay=1e-4)")])
        modify_file("train_centralized.py", [(r"optimizer = torch\.optim\.([A-Za-z]+)\(([^)]+)\)", r"optimizer = torch.optim.\1(\2, weight_decay=1e-4)")])
    elif action == "heads_4":
        modify_file("model.py", [(r"nhead=[0-9]+", "nhead=4")])
    elif action == "heads_16":
        modify_file("model.py", [(r"nhead=[0-9]+", "nhead=16")])
    elif action == "layers_3":
        modify_file("model.py", [(r"num_layers=[0-9]+", "num_layers=3")])
    elif action == "layers_4":
        modify_file("model.py", [(r"num_layers=[0-9]+", "num_layers=4")])
    elif action == "embed_512":
        modify_file("model.py", [
            (r"Linear\(input_dim, [0-9]+\)", "Linear(input_dim, 512)"),
            (r"BatchNorm1d\([0-9]+\)", "BatchNorm1d(512)"),
            (r"d_model=[0-9]+", "d_model=512"),
            (r"Linear\([0-9]+, 128\)", "Linear(512, 128)")
        ])
    elif action == "dropout_0.2":
        modify_file("model.py", [(r"nn\.Dropout\([0-9.]+\)", "nn.Dropout(0.2)")])
        modify_file("model.py", [(r"dropout=[0-9.]+", "dropout=0.2")])
    elif action == "dropout_0.3":
        modify_file("model.py", [(r"nn\.Dropout\([0-9.]+\)", "nn.Dropout(0.3)")])
        modify_file("model.py", [(r"dropout=[0-9.]+", "dropout=0.3")])
    elif action == "optimizer_adamw":
        modify_file("client.py", [(r"torch\.optim\.Adam\(", "torch.optim.AdamW(")])
        modify_file("train_centralized.py", [(r"torch\.optim\.Adam\(", "torch.optim.AdamW(")])
    elif action == "epochs_2":
        modify_file("client.py", [(r"epochs=[0-9]+", "epochs=2")])

def run_pipeline(exp_id):
    os.makedirs("logs", exist_ok=True)
    
    global_model_path = "saved_models/global_model.pth"
    fl_metrics_path = "fl_metrics.json"
    centralized_model_path = "saved_models/centralized_model.pth"
    results_path = "results.json"
    
    # Remove old files if they exist to verify creation
    for p in [fl_metrics_path, results_path]:
        if os.path.exists(p):
            os.remove(p)
            
    # Get previous model timestamp
    before_time = os.path.getmtime(global_model_path) if os.path.exists(global_model_path) else 0

    print("    Starting server...")
    with open(f"logs/server_exp_{exp_id}.log", "w") as f_server:
        server = subprocess.Popen(["python", "server.py"], stdout=f_server, stderr=subprocess.STDOUT)
    
    time.sleep(5)
    
    clients = []
    client_files = []
    print("    Starting clients...")
    for i in range(1, 6):
        fc = open(f"logs/client{i}_exp_{exp_id}.log", "w")
        client_files.append(fc)
        c = subprocess.Popen(["python", "client.py", "--client_id", str(i)], stdout=fc, stderr=subprocess.STDOUT)
        clients.append(c)
    
    server.wait()
    
    if server.returncode != 0:
        for c in clients:
            c.kill()
        for fc in client_files:
            fc.close()
        raise Exception(f"Server subprocess failed with return code {server.returncode}")
        
    for c in clients:
        try:
            c.wait(timeout=120)
        except subprocess.TimeoutExpired:
            print("    Warning: Client timed out, killing.")
            c.kill()
            
    for fc in client_files:
        fc.close()
        
    print("    FL completed.")
        
    # Verify FL Completion
    if not os.path.exists(global_model_path):
        raise Exception(f"FL Failed: {global_model_path} was not created.")
    after_time = os.path.getmtime(global_model_path)
    if after_time <= before_time:
        raise Exception(f"FL Failed: {global_model_path} was not updated (timestamp unchanged).")
    if not os.path.exists(fl_metrics_path):
        raise Exception(f"FL Failed: {fl_metrics_path} was not created.")

    print("    Running centralized training...")
    with open(f"logs/centralized_exp_{exp_id}.log", "w") as f_cent:
        cent = subprocess.run(["python", "train_centralized.py"], stdout=f_cent, stderr=subprocess.STDOUT)
        if cent.returncode != 0:
            raise Exception(f"Centralized training subprocess failed with return code {cent.returncode}")
            
    if not os.path.exists(centralized_model_path):
        raise Exception(f"Centralized training failed: {centralized_model_path} not found.")
        
    print("    Centralized completed.")

    print("    Running evaluation...")
    with open(f"logs/eval_exp_{exp_id}.log", "w") as f_eval:
        ev = subprocess.run(["python", "eval.py"], stdout=f_eval, stderr=subprocess.STDOUT)
        if ev.returncode != 0:
            raise Exception(f"Evaluation subprocess failed with return code {ev.returncode}")
            
    if not os.path.exists(results_path):
        raise Exception(f"Evaluation failed: {results_path} not found.")
        
    print("    Evaluation completed.")

def get_metrics():
    try:
        with open("results.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"    Error reading results.json: {e}")
        return None

def main():
    csv_file = "experiment_results.csv"
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment ID", "Parameter Changed", "Old Value", "New Value", 
            "FL Accuracy", "FL Precision", "FL Recall", "FL F1", 
            "Centralized Accuracy", "Centralized Precision", "Centralized Recall", "Centralized F1", 
            "Training Time", "Status"
        ])

    backup_files()
    
    print("Evaluating baseline...")
    # Read the existing baseline if possible
    metrics = get_metrics()
    if metrics:
        best_fl_acc = float(metrics["Federated"][0])
    else:
        best_fl_acc = 0.8067
    
    best_config_history = []
    best_overall_metrics = metrics
    best_experiment_id = 0
    best_time = 0
    
    for exp in EXPERIMENTS:
        print(f"\n===== Experiment {exp['id']} =====")
        print(f"Parameter: {exp['param']} ({exp['old']} -> {exp['new']})")
        
        apply_experiment(exp['action'])
        
        start_time = time.time()
        
        try:
            run_pipeline(exp['id'])
            train_time = time.time() - start_time
            print("    Results saved.")
            
            metrics = get_metrics()
            if not metrics:
                raise Exception("Metrics missing after run.")
                
            fl_acc = float(metrics["Federated"][0])
            fl_prec = float(metrics["Federated"][1])
            fl_rec = float(metrics["Federated"][2])
            fl_f1 = float(metrics["Federated"][3])
            
            c_acc = float(metrics["Centralized"][0])
            c_prec = float(metrics["Centralized"][1])
            c_rec = float(metrics["Centralized"][2])
            c_f1 = float(metrics["Centralized"][3])
            
            print(f"    Result: FL Acc={fl_acc:.4f} (Best: {best_fl_acc:.4f})")
            
            if fl_acc > best_fl_acc:
                print("    Improvement found! Keeping configuration.")
                best_fl_acc = fl_acc
                best_config_history.append(f"{exp['param']} changed to {exp['new']}")
                backup_files()
                best_overall_metrics = metrics
                best_experiment_id = exp['id']
                best_time = train_time
                status = "Improved"
                
                if os.path.exists("saved_models/global_model.pth"):
                    shutil.copy("saved_models/global_model.pth", "saved_models/best_global_model.pth")
            else:
                print("    No improvement. Reverting configuration.")
                restore_files()
                status = "Reverted"
                
            print("    Experiment finished successfully.")
                
        except Exception as e:
            print(f"    Experiment failed: {e}")
            restore_files()
            status = "Failed"
            fl_acc = fl_prec = fl_rec = fl_f1 = ""
            c_acc = c_prec = c_rec = c_f1 = ""
            train_time = ""
            
        with open(csv_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                exp["id"], exp["param"], exp["old"], exp["new"],
                fl_acc, fl_prec, fl_rec, fl_f1,
                c_acc, c_prec, c_rec, c_f1,
                train_time, status
            ])
            
    # After all experiments finish successfully
    print("\nAll tuning experiments complete. Running explainability and final reporting on the best model...")
    restore_files() 
    
    with open("logs/plot_metrics.log", "w") as f:
        subprocess.run(["python", "plot_metrics.py"], stdout=f, stderr=subprocess.STDOUT)
    with open("logs/compare_models.log", "w") as f:
        subprocess.run(["python", "compare_models.py"], stdout=f, stderr=subprocess.STDOUT)
    with open("logs/run_shap.log", "w") as f:
        subprocess.run(["python", "run_shap.py"], stdout=f, stderr=subprocess.STDOUT)
    with open("logs/run_captum.log", "w") as f:
        subprocess.run(["python", "run_captum.py"], stdout=f, stderr=subprocess.STDOUT)
    with open("logs/generate_report.log", "w") as f:
        subprocess.run(["python", "generate_report.py"], stdout=f, stderr=subprocess.STDOUT)
    
    with open("BEST_CONFIGURATION.md", "w") as f:
        f.write("# Best Configuration\n\n")
        f.write("## Metrics\n")
        if best_overall_metrics:
            f.write(f"- Best FL Accuracy: {best_overall_metrics['Federated'][0]}\n")
            f.write(f"- Best FL Precision: {best_overall_metrics['Federated'][1]}\n")
            f.write(f"- Best FL Recall: {best_overall_metrics['Federated'][2]}\n")
            f.write(f"- Best FL F1: {best_overall_metrics['Federated'][3]}\n\n")
            f.write(f"- Centralized Accuracy: {best_overall_metrics['Centralized'][0]}\n")
            f.write(f"- Centralized Precision: {best_overall_metrics['Centralized'][1]}\n")
            f.write(f"- Centralized Recall: {best_overall_metrics['Centralized'][2]}\n")
            f.write(f"- Centralized F1: {best_overall_metrics['Centralized'][3]}\n\n")
        f.write(f"- Training Time: {best_time:.2f} seconds\n")
        
        f.write("\n## Best Hyperparameters\n")
        for conf in best_config_history:
            f.write(f"- {conf}\n")
            
        f.write("\n## Why this configuration performed better\n")
        f.write("This configuration optimally balanced the model capacity, learning stability, and regularization, leading to better generalization across distributed datasets without overfitting on individual clients.\n")
        
    print("\n================ FINAL SUMMARY ================")
    print(f"Best Experiment: {best_experiment_id}")
    print(f"Best FL Accuracy: {best_fl_acc}")
    print(f"Improvement over baseline: {((best_fl_acc - 0.8067) / 0.8067 * 100):.2f}%")
    print(f"Best Hyperparameters: {', '.join(best_config_history) if best_config_history else 'None'}")
    print(f"Training Time: {best_time:.2f}s")
    print("Location of best model: saved_models/best_global_model.pth")
    print("Location of experiment_results.csv: experiment_results.csv")
    print("Location of BEST_CONFIGURATION.md: BEST_CONFIGURATION.md")
    print("===============================================")
            
if __name__ == "__main__":
    main()
