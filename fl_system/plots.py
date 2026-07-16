import matplotlib.pyplot as plt
import numpy as np

def plot_fl_accuracy_vs_rounds(rounds, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    plt.title('Federated Learning: Accuracy vs. Communication Rounds')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('fl_accuracy_vs_rounds.png')
    print("Saved plot to fl_accuracy_vs_rounds.png")

def plot_centralized_vs_fl(metrics_labels, centralized_scores, fl_scores):
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, centralized_scores, width, label='Centralized', color='royalblue')
    rects2 = ax.bar(x + width/2, fl_scores, width, label='Federated', color='darkorange')
    
    ax.set_ylabel('Scores')
    ax.set_title('Centralized vs. Federated Learning Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('centralized_vs_fl_bar_chart.png')
    print("Saved plot to centralized_vs_fl_bar_chart.png")

import os
import json

def main():
    if os.path.exists("fl_metrics.json"):
        with open("fl_metrics.json", "r") as f:
            accuracies = json.load(f)
        rounds = list(range(1, len(accuracies) + 1))
        plot_fl_accuracy_vs_rounds(rounds, accuracies)
    else:
        print("Warning: fl_metrics.json not found. Run server.py first to generate FL metrics.")
        
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            results = json.load(f)
        metrics_labels = results["Metric"]
        centralized_scores = [float(x) for x in results["Centralized"]]
        fl_scores = [float(x) for x in results["Federated"]]
        plot_centralized_vs_fl(metrics_labels, centralized_scores, fl_scores)
    else:
        print("Warning: results.json not found. Run eval.py first to generate evaluation metrics.")

if __name__ == "__main__":
    main()
