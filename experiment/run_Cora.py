import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops
import time
import psutil
from memory_profiler import memory_usage
import numpy as np
import matplotlib.pyplot as plt

from models.GCN import GCN
from models.MLP import MLP
from poisoning.poisoning import apply_poisoning
from defense.pruning import prune_and_restore_edges
from defense.bayesian import bayesian_predict
from defense.anomaly import detect_anomalous_nodes
from defense.smoothing import randomized_smoothing
from utils.train_eval import train, evaluate, calculate_asr, measure_resources
from utils.data_utils import load_data

def get_model(model_name, in_channels, hidden_channels, out_channels):
    if model_name.lower() == 'gcn':
        return GCN(in_channels, hidden_channels, out_channels)
    elif model_name.lower() == 'mlp':
        return MLP(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_experiment(model_name, poisoning_rate, target_class):
    # Load clean data
    dataset, data = load_data(name='Cora', root_dir='data/Cora')
    hidden_dim = 16
    num_classes = dataset.num_classes
    num_features = data.num_features
    results = {}
    is_mlp = model_name.lower() == 'mlp'

    # --- 1. Train on Clean Data ---
    clean_model = get_model(model_name, num_features, hidden_dim, num_classes)
    if is_mlp:
        clean_optim = torch.optim.Adam(clean_model.parameters(), lr=0.01)
    else:
        clean_optim = torch.optim.Adam(clean_model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        if is_mlp:
            clean_model.train()
            clean_optim.zero_grad()
            out = clean_model(data.x)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            clean_optim.step()
        else:
            train(clean_model, data, clean_optim)

    # Evaluate clean model
    if is_mlp:
        clean_model.eval()
        with torch.no_grad():
            out = clean_model(data.x)
            pred = out.argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            clean_acc = (correct / data.test_mask.sum()).item()
    else:
        clean_acc = evaluate(clean_model, data)
    results['clean_acc'] = clean_acc

    # --- 2. Apply Poisoning and Train ---
    poisoned_data = apply_poisoning(data.clone(), num_classes, target_class, poisoning_rate)
    poisoned_model = get_model(model_name, num_features, hidden_dim, num_classes)
    
    if is_mlp:
        poisoned_optim = torch.optim.Adam(poisoned_model.parameters(), lr=0.01)
    else:
        poisoned_optim = torch.optim.Adam(poisoned_model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Train poisoned model
    for epoch in range(200):
        if is_mlp:
            poisoned_model.train()
            poisoned_optim.zero_grad()
            out = poisoned_model(poisoned_data.x)
            loss = F.cross_entropy(out[poisoned_data.train_mask], poisoned_data.y[poisoned_data.train_mask])
            loss.backward()
            poisoned_optim.step()
        else:
            train(poisoned_model, poisoned_data, poisoned_optim)

    # Evaluate poisoned model
    if is_mlp:
        poisoned_model.eval()
        with torch.no_grad():
            out = poisoned_model(poisoned_data.x)
            pred = out.argmax(dim=1)
            if poisoned_data.test_mask.sum() == 0:
                 poisoned_acc = float('nan')
            else:
                 correct = (pred[poisoned_data.test_mask] == poisoned_data.y[poisoned_data.test_mask]).sum()
                 poisoned_acc = (correct / poisoned_data.test_mask.sum()).item()
            
            target_mask = (poisoned_data.y == target_class) & poisoned_data.test_mask
            if target_mask.sum() == 0:
                asr_poisoned = float('nan')
            else:
                misclassified = (pred[target_mask] != poisoned_data.y[target_mask]).sum()
                asr_poisoned = (misclassified / target_mask.sum()).item()
    else:
        poisoned_acc = evaluate(poisoned_model, poisoned_data)
        asr_poisoned = calculate_asr(poisoned_model, poisoned_data, target_class)
    
    results['poisoned_acc'] = poisoned_acc
    results['asr_poisoned'] = asr_poisoned

    # --- 3. Apply Defense ---
    
    # Step 1: Pruning (Structural defense, applied to all)
    defended_data = prune_and_restore_edges(poisoned_data.clone())
    
    if defended_data.test_mask.sum() == 0:
        results['defended_acc'] = float('nan')
        results['asr_defended'] = float('nan')
        results['cad'] = float('nan')
        return results

    # --- Defense for MLP (Pruning only) ---
    if is_mlp:
        defended_model = get_model(model_name, num_features, hidden_dim, num_classes)
        defended_optim = torch.optim.Adam(defended_model.parameters(), lr=0.01)
        for epoch in range(200):
            defended_model.train()
            defended_optim.zero_grad()
            out = defended_model(defended_data.x)
            loss = F.cross_entropy(out[defended_data.train_mask], defended_data.y[defended_data.train_mask])
            loss.backward()
            defended_optim.step()

        # Evaluate defended MLP
        defended_model.eval()
        with torch.no_grad():
            final_preds = defended_model(defended_data.x).argmax(dim=1)
            if defended_data.test_mask.sum() == 0:
                defended_acc_val = float('nan')
            else:
                correct = (final_preds[defended_data.test_mask] == defended_data.y[defended_data.test_mask]).sum()
                defended_acc_val = (correct / defended_data.test_mask.sum()).item()
            
            # ASR for defended MLP
            target_mask = (defended_data.y == target_class) & defended_data.test_mask
            if target_mask.sum() == 0:
                asr_defended = float('nan')
            else:
                misclassified = (final_preds[target_mask] != defended_data.y[target_mask]).sum()
                asr_defended = (misclassified / target_mask.sum()).item()
        
        results['defended_acc'] = defended_acc_val
        results['asr_defended'] = asr_defended

    # --- Hybrid Defense (for GNNs) ---
    else:
        # Step 2: Train base GNN (specified by model_name)
        gnn_model = get_model(model_name, num_features, hidden_dim, num_classes)
        gnn_optim = torch.optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=5e-4)
        for epoch in range(200):
            train(gnn_model, defended_data, gnn_optim, defended_data.edge_index)

        # Step 3: Anomaly detection on embeddings
        with torch.no_grad():
            embeddings = gnn_model(defended_data.x, defended_data.edge_index)
        anomalous_nodes = detect_anomalous_nodes(embeddings)
        
        # Step 4: Train MLP component
        mlp = MLP(defended_data.x.size(1), dataset.num_classes)
        mlp_optim = torch.optim.Adam(mlp.parameters(), lr=0.01)
        for epoch in range(100):
            mlp.train()
            pred = mlp(defended_data.x)
            loss = F.cross_entropy(pred[defended_data.train_mask], defended_data.y[defended_data.train_mask])
            mlp_optim.zero_grad()
            loss.backward()
            mlp_optim.step()

        # Step 5: Get predictions from all components
        smoothed_preds = randomized_smoothing(gnn_model, defended_data.x, defended_data.edge_index)
        gnn_preds = gnn_model(defended_data.x, defended_data.edge_index).argmax(dim=1)
        mlp_preds = mlp(defended_data.x).argmax(dim=1)
        
        # Hybrid prediction: Trust consensus, else use smoothing
        final_preds = torch.where(gnn_preds == mlp_preds, gnn_preds, smoothed_preds)
        
        # Create defended test mask (excluding anomalies)
        test_mask_defended = defended_data.test_mask.clone()
        test_mask_defended[anomalous_nodes] = False
        
        if test_mask_defended.sum() == 0:
            defended_acc_val = float('nan')
        else:
            correct = (final_preds[test_mask_defended] == defended_data.y[test_mask_defended]).sum()
            defended_acc_val = (correct / test_mask_defended.sum()).item()
        
        # Calculate ASR on the *original* test mask
        asr_defended = calculate_asr(lambda x, ei: final_preds, defended_data, target_class)

        results['defended_acc'] = defended_acc_val
        results['asr_defended'] = asr_defended

    results['cad'] = clean_acc - results['defended_acc'] # Clean Accuracy Drop
    return results


if __name__ == "__main__":
    model_names_to_run = ['gcn', 'mlp']
    poisoning_rates = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80]
    target_class = 0
    
    # Use a dictionary to store results per model
    all_results = {model_name: [] for model_name in model_names_to_run}
    resource_usage = {model_name: [] for model_name in model_names_to_run}

    # Loop over models, then rates
    for model_name in model_names_to_run:
        print(f"\n--- Starting experiments for model: {model_name.upper()} ---")
        for rate in poisoning_rates:
            print(f"\n=== Running {model_name.upper()} with poisoning rate: {rate*100:.1f}% ===")
            
            # Measure resources and get results simultaneously
            mem_usage, (time_used, cpu_used, results) = memory_usage((lambda mn=model_name, r=rate, tc=target_class: (
                *measure_resources(run_experiment, mn, r, tc)[0:2], # (time, cpu)
                run_experiment(mn, r, tc) # (results)
            )), retval=True, max_usage=True, interval=0.1)

            all_results[model_name].append((rate, results))
            resource_usage[model_name].append((rate, time_used, mem_usage, cpu_used))
            
            print(f"Clean Accuracy: {results['clean_acc']:.4f}")
            print(f"Poisoned Accuracy: {results['poisoned_acc']:.4f} (ASR: {results['asr_poisoned']:.4f})")
            print(f"Defended Accuracy: {results['defended_acc']:.4f} (ASR: {results['asr_defended']:.4f})")
            print(f"Clean Accuracy Drop (CAD): {results['cad']:.4f}")
            print(f"Time = {time_used:.2f}s | Memory = {mem_usage:.2f} MiB | CPU = {cpu_used:.2f}%")

    print("\nGenerating comparative plots...")
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Defended Accuracy
    plt.subplot(2, 2, 1)
    for model_name in model_names_to_run:
        rates = [r[0] for r in all_results[model_name]]
        defended_accs = [r[1]['defended_acc'] for r in all_results[model_name]]
        plt.plot(rates, defended_accs, label=f'{model_name.upper()} Defended', marker='o', alpha=0.7)
    
    # Plot clean GCN as baseline
    rates_gcn = [r[0] for r in all_results['gcn']]
    clean_accs_gcn = [r[1]['clean_acc'] for r in all_results['gcn']]
    plt.plot(rates_gcn, clean_accs_gcn, 'k--', label='GCN Clean (Baseline)', alpha=0.8)
    plt.xlabel('Poisoning Rate')
    plt.ylabel('Defended Accuracy')
    plt.legend()
    plt.title('Defended Accuracy vs. Poisoning Rate')
    plt.grid(True)

    # Plot 2: Attack Success Rate (ASR)
    plt.subplot(2, 2, 2)
    for model_name in model_names_to_run:
        rates = [r[0] for r in all_results[model_name]]
        asr_poisoned = [r[1]['asr_poisoned'] for r in all_results[model_name]]
        asr_defended = [r[1]['asr_defended'] for r in all_results[model_name]]
        plt.plot(rates, asr_poisoned, label=f'{model_name.upper()} Poisoned ASR', marker='x', linestyle=':', alpha=0.5)
        plt.plot(rates, asr_defended, label=f'{model_name.upper()} Defended ASR', marker='s', alpha=0.8)
    
    plt.xlabel('Poisoning Rate')
    plt.ylabel('Attack Success Rate (ASR)')
    plt.legend(fontsize='small')
    plt.title('ASR (Poisoned vs. Defended) vs. Poisoning Rate')
    plt.grid(True)

    # Plot 3: Clean Accuracy Drop (CAD)
    plt.subplot(2, 2, 3)
    for model_name in model_names_to_run:
        rates = [r[0] for r in all_results[model_name]]
        cads = [r[1]['cad'] for r in all_results[model_name]]
        plt.plot(rates, cads, label=f'{model_name.upper()} CAD', marker='^', alpha=0.7)
    plt.xlabel('Poisoning Rate')
    plt.ylabel('Clean Accuracy Drop (Clean Acc - Defended Acc)')
    plt.legend()
    plt.title('Clean Accuracy Drop vs. Poisoning Rate')
    plt.grid(True)

    # Plot 4: Resource Usage (Time)
    plt.subplot(2, 2, 4)
    for model_name in model_names_to_run:
        rates = [r[0] for r in resource_usage[model_name]]
        times = [r[1] for r in resource_usage[model_name]]
        plt.plot(rates, times, label=f'{model_name.upper()} Time', marker='d', alpha=0.7)
    plt.xlabel('Poisoning Rate')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.title('Experiment Runtimes')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/defense_comparison_results.png')
    print("\nComparative results plot saved to results/defense_comparison_results.png")
    plt.show()
