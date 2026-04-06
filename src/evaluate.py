# calculates recall@10 and NDCG@10 metrics 
from dataset import build_data
from model import gmf_lay, mlp_lay, neumf_lay
import os
import numpy  as np
import torch
import json
import matplotlib.pyplot as plt

# Configuration for each model to evaluate
configs = [
    {
        "name" : "MLP  [64,32,16]  emb=32",
        "ratings_path" : "./data/ml-1m/ratings.dat",
        "batch_size" : 256,
        "neg_ratio" : 4, 
        "model_type" : "mlp", 
        "num_factors" : 32,  # embedding dimension
        "layers" : [64, 32, 16],  # hidden layer sizes
        "checkpoint_path" : "checkpoints/mlp_32.pt",  # trained model weights
        "his_path" : "results/train_mlp_32.json",  # training history for plotting
        "seed" : 42,
    },
    {
        "name" : "GMF-only emb=32",
        "ratings_path" : "./data/ml-1m/ratings.dat",
        "batch_size" : 256,
        "neg_ratio" : 4,
        "model_type" : "gmf",  
        "num_factors" : 32,
        "layers" : [64, 32, 16],
        "checkpoint_path" : "checkpoints/gmf_32.pt",
        "his_path" : "results/train_gmf_32.json",
        "seed" : 42,
    },
    {
        "name" : "NeuMF [64,32,16]  emb=32  (pretrained)",
        "ratings_path" : "./data/ml-1m/ratings.dat",
        "batch_size" : 256,
        "neg_ratio" : 4,
        "model_type" : "neumf",
        "num_factors" : 32,
        "layers" : [64, 32, 16],
        "checkpoint_path" : "checkpoints/neumf_32.pt",
        "his_path" : "results/train_neumf_32.json",
        "seed" : 42,
    },
]

def eval():
# load the data and build dataloaders
    print("\nloading data")
    train_loader, val_loader, test_loader, num_users, num_items, user_history = build_data(
        filepath = "./data/ml-1m/ratings.dat",
        batch_size = 256,
        neg_ratio = 4,
        seed = 42,
    )
# build ground truth for map each user
    ground_truth = {}
    for user_ids, item_ids, labels in test_loader:
        for u, i, l in zip(user_ids, item_ids, labels):
            if l.item() == 1:  # only positive interactions 
                if u.item() not in ground_truth:
                    ground_truth[u.item()] = set()
                ground_truth[u.item()].add(i.item())

# all possible items
    all_items = set(range(num_items))
    results = []  # store evaluation results for each model
# evaluate each configured model
    for config in configs:
        model_type = config["model_type"]
        print(f"\n{'─' * 55}")
        print(f"Evaluating : {config['name']}")
        print(f"Checkpoint : {config['checkpoint_path']}")
# instantiate the appropriate model architecture
        if model_type == "neumf":
            model = neumf_lay(
                num_users = num_users,
                num_items = num_items,
                num_factors = config["num_factors"],
                layers = config["layers"],
            )
        elif model_type == "gmf":
            model = gmf_lay(
                num_users = num_users,
                num_items = num_items,
                num_factors = config["num_factors"],
            )
        elif model_type == "mlp":
            model = mlp_lay(
                num_users = num_users,
                num_items = num_items,
                layers = config["layers"],
            )
        else:
            raise ValueError(
                f"unknown model type: {model_type}. use 'neumf', 'gmf', or 'mlp'"
            )
        
# displaying model size
        total_params = sum(p.numel() for p in model.parameters())
        print(f"parameters : {total_params:,}")
        
# load trained weights
        model.load_state_dict(torch.load(config["checkpoint_path"], map_location="cpu"))
        model.eval()
# Compute metrics for each user
        recall_scores = []
        ndcg_scores = []
        for user_id, true_items in ground_truth.items():
            # get items the user has already seen 
            seen_items = user_history.get(user_id, set())
            # candidate items, all items except those already seen
            candidates = list(all_items - seen_items)
            # Creating tensors repeat user_id for each candidate item
            user_tensor = torch.tensor([user_id]).repeat(len(candidates))
            candidate_tensor = torch.tensor(candidates)
            # score all candidate items for this user
            with torch.no_grad():
                scores = model(user_tensor, candidate_tensor)
            # Get top 10 highest scored items
            top_indices = torch.topk(scores, 10, largest=True).indices
            top_items = [candidates[idx] for idx in top_indices]
            # recall@10
            hits = len(set(top_items) & true_items)
            recall = hits / len(true_items)
            recall_scores.append(recall)
            # NDCG@10
            dcg = 0.0
            for rank, item in enumerate(top_items):
                if item in true_items:
                    dcg += 1/np.log2(rank + 2)  # rank+2 because rank is 0-indexed
            # Ideal DCG
            ideal_dcg = 0.0
            for rank in range(min(len(true_items), 10)):
                ideal_dcg += 1/np.log2(rank + 2)
            # normalize DCG by ideal DCG
            ndcg = dcg/ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        # Average metrics across all users
        final_recall = np.mean(recall_scores)
        final_ndcg = np.mean(ndcg_scores)
        print(f"Recall@10 : {final_recall:.4f}")
        print(f"NDCG@10 : {final_ndcg:.4f}")
        # store results for this model
        results.append({
            "name" : config["name"],
            "recall" : round(final_recall, 4),
            "ndcg" : round(final_ndcg, 4),
            "params" : total_params,
        })
# display results table
    print(f"\n{'=' * 60}")
    print("results table recall@10 and ndcg@10")
    print(f"{'=' * 60}")
    print(f"{'configuration':<42} {'recall@10':>10} {'ndcg@10':>10}")
    print(f"{'─' * 42} {'─' * 10} {'─' * 10}")
    for r in results:
        print(f"{r['name']:<42} {r['recall']:>10.4f} {r['ndcg']:>10.4f}")
    print(f"{'=' * 60}")
    # save results to JSON
    os.makedirs("results", exist_ok=True)
    with open("./results/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nresults saved results/eval_results.json")
    # load training histories for loss curve plotting
    curves = []
    for config in configs:
        hp = config.get("his_path", "")
        if hp and os.path.exists(hp):
            with open(hp) as f:
                curves.append((config["name"], json.load(f)))
    # Color scheme for charts
    curve_colors = ["#2166AC", "#1A9850", "#D6604D"]
    # plot training/validation loss curves
    if curves:
        fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4.5),sharey=False)
        if len(curves) == 1:
            axes = [axes]  
        for idx, (ax, (name, hist)) in enumerate(zip(axes, curves)):
            color = curve_colors[idx % len(curve_colors)]
            epochs_ran = list(range(1, len(hist["train_loss"]) + 1))
            # find epoch with best validation loss
            best_ep = int(np.argmin(hist["val_loss"])) + 1
            # plot train and validation loss
            ax.plot(epochs_ran, hist["train_loss"], color=color, linewidth=2, marker="o", markersize=3, label="Train Loss")
            ax.plot(epochs_ran, hist["val_loss"], color=color, linewidth=2, marker="o", markersize=3, linestyle="--", alpha=0.75, label="Val Loss")
            # mark best validation 
            ax.axvline(x=best_ep, color="grey", linestyle=":", linewidth=1.2, alpha=0.8)
            ax.text(best_ep + 0.15, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.16,"best\nval", fontsize=7, color="grey", va="bottom")
            # Formatting
            ax.set_title(name, fontsize=9, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("BCE Loss", fontsize=9)
            ax.set_xlim(left=1)
            ax.set_ylim(bottom=0.15)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)
        fig.suptitle("Training and Validation Loss — NCF Models", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("./results/loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("loss curves saved results/loss_curves.png")
# generate bar charts comparing model performance
    if results:
# create short names for x-axis labels
        short_names = []
        for r in results:
            n = r["name"]
            if n.startswith("MLP"):
                short_names.append("MLP")
            elif n.startswith("GMF"):
                short_names.append("GMF")
            else:
                short_names.append("NeuMF")
# extract metric values
        recalls = [r["recall"] for r in results]
        ndcgs = [r["ndcg"] for r in results]
        bar_colors = ["#2166AC", "#1A9850", "#D6604D"]
# Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# panel 1 Recall@10 comparison
        ax = axes[0]
        bars = ax.bar(short_names, recalls, color=bar_colors[:len(results)], width=0.5, zorder=3)
# add value labels on top of bars
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
# add minimum threshold line
        ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1, label="Min threshold (0.05)")
        ax.set_title("recall@10 comparison", fontsize=11, fontweight="bold")
        ax.set_ylabel("recall@10", fontsize=10)
        ax.set_ylim(0, max(recalls) + 0.04)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
# panel 2 ndcg@10 comparison
        ax = axes[1]
        bars = ax.bar(short_names, ndcgs, color=bar_colors[:len(results)], width=0.5, zorder=3)
# add value labels on top of bars
        for bar, val in zip(bars, ndcgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
# add minimum threshold line
        ax.axhline(y=0.03, color="red", linestyle="--", linewidth=1, label="Min threshold (0.03)")
        ax.set_title("ndcg@10 comparison", fontsize=11, fontweight="bold")
        ax.set_ylabel("ndcg@10", fontsize=10)
        ax.set_ylim(0, max(ndcgs) + 0.06)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
# panel 3 Side-by-side comparison
        ax = axes[2]
        x = np.arange(len(results))
        width = 0.35
# create grouped bars recall and ndcg side by side
        bars_r = ax.bar(x - width / 2, recalls, width, color=bar_colors[:len(results)], label="Recall@10", zorder=3)
        bars_n = ax.bar(x + width / 2, ndcgs, width, color=bar_colors[:len(results)], alpha=0.45, hatch="///", label="NDCG@10", zorder=3)
# add value labels
        for bar, val in zip(bars_r, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars_n, ndcgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
# formatting
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_title("recall@10 vs ndcg@10", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0, max(max(recalls), max(ndcgs)) + 0.06)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        fig.suptitle("ncf model comparison movieLens 1M", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("./results/bar_charts.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("bar charts saved results/bar_charts.png")
# Entry point
if __name__ == "__main__":
    eval()