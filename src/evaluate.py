""" Evaluate the performance using Recall@10, NDCG@10. """

from dataset import build_dataloaders
from model   import gmf_lay, mlp_lay, neumf_lay

import os
import numpy  as np
import torch
import json
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Add or remove configs here to match whichever checkpoints you have trained.
# The "name" field is only used for the printed results table.
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = [
    {
        "name"           : "MLP  [64,32,16]  emb=32",
        "ratings_path"   : "./data/ml-1m/ratings.dat",
        "batch_size"     : 256,
        "neg_ratio"      : 4,
        "model_type"     : "mlp",
        "num_factors"    : 32,
        "layers"         : [64, 32, 16],
        "checkpoint_path": "checkpoints/mlp_32.pt",
        "history_path"   : "results/train_mlp_32.json",
        "seed"           : 42,
    },
    {
        "name"           : "GMF-only         emb=32",
        "ratings_path"   : "./data/ml-1m/ratings.dat",
        "batch_size"     : 256,
        "neg_ratio"      : 4,
        "model_type"     : "gmf",
        "num_factors"    : 32,
        "layers"         : [64, 32, 16],
        "checkpoint_path": "checkpoints/gmf_32.pt",
        "history_path"   : "results/train_gmf_32.json",
        "seed"           : 42,
    },
    {
        "name"           : "NeuMF [64,32,16]  emb=32  (pretrained)",
        "ratings_path"   : "./data/ml-1m/ratings.dat",
        "batch_size"     : 256,
        "neg_ratio"      : 4,
        "model_type"     : "neumf",
        "num_factors"    : 32,
        "layers"         : [64, 32, 16],
        "checkpoint_path": "checkpoints/neumf_32.pt",
        "history_path"   : "results/train_neumf_32.json",
        "seed"           : 42,
    },
    # ── Optional 4th config ───────────────────────────────────────────────────
    # Uncomment only if you have trained and saved this checkpoint.
    # {
    #     "name"           : "NeuMF [128,64,32] emb=64  (pretrained)",
    #     "ratings_path"   : "./data/ml-1m/ratings.dat",
    #     "batch_size"     : 256,
    #     "neg_ratio"      : 4,
    #     "model_type"     : "neumf",
    #     "num_factors"    : 64,
    #     "layers"         : [128, 64, 32],
    #     "checkpoint_path": "checkpoints/neumf_128_64_32.pt",
    #     "history_path"   : "results/train_neumf_128_64_32.json",
    #     "seed"           : 42,
    # },
]


def evaluate():
    print("\nLoading data")
    train_loader, val_loader, test_loader, num_users, num_items, user_history = build_dataloaders(
        filepath   = "./data/ml-1m/ratings.dat",
        batch_size = 256,
        neg_ratio  = 4,
        seed       = 42,
    )

    # Build ground truth: for each user, collect test positive item IDs.
    # test_loader contains positives only (no negatives) from dataset.py.
    ground_truth = {}
    for user_ids, item_ids, labels in test_loader:
        for u, i, l in zip(user_ids, item_ids, labels):
            if l.item() == 1:
                if u.item() not in ground_truth:
                    ground_truth[u.item()] = set()
                ground_truth[u.item()].add(i.item())

    all_items = set(range(num_items))

    # Collect results for the summary table
    results = []

    for CONFIG in CONFIGS:
        model_type = CONFIG["model_type"]
        print(f"\n{'─' * 55}")
        print(f"Evaluating : {CONFIG['name']}")
        print(f"Checkpoint : {CONFIG['checkpoint_path']}")

        if model_type == "neumf":
            model = neumf_lay(
                num_users   = num_users,
                num_items   = num_items,
                num_factors = CONFIG["num_factors"],
                layers      = CONFIG["layers"],
            )
        elif model_type == "gmf":
            model = gmf_lay(
                num_users   = num_users,
                num_items   = num_items,
                num_factors = CONFIG["num_factors"],
            )
        elif model_type == "mlp":
            model = mlp_lay(
                num_users = num_users,
                num_items = num_items,
                layers    = CONFIG["layers"],
            )
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Use 'neumf', 'gmf', or 'mlp'."
            )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters : {total_params:,}")

        # Load the final fine-tuned checkpoint directly.
        # load_pretrained_weights() is only called during training (train.py).
        # Here we always load the saved checkpoint which already contains the
        # fully fine-tuned weights — no pretrain step needed at eval time.
        model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location="cpu"))
        model.eval()

        recall_scores = []
        ndcg_scores   = []

        # Full-ranking evaluation: for each test user score all unseen items.
        # user_history contains train+val positives only (not test positives),
        # so test positives remain as valid candidates — correct masking.
        for user_id, true_items in ground_truth.items():
            seen_items       = user_history.get(user_id, set())
            candidates       = list(all_items - seen_items)
            user_tensor      = torch.tensor([user_id]).repeat(len(candidates))
            candidate_tensor = torch.tensor(candidates)

            with torch.no_grad():
                scores = model(user_tensor, candidate_tensor)

            # Sort descending and take top 10
            top_indices = torch.topk(scores, 10, largest=True).indices
            top_items   = [candidates[idx] for idx in top_indices]

            # Recall@10
            hits   = len(set(top_items) & true_items)
            recall = hits / len(true_items)
            recall_scores.append(recall)

            # NDCG@10 — rank is 0-indexed so denominator is log2(rank+2)
            dcg = 0.0
            for rank, item in enumerate(top_items):
                if item in true_items:
                    dcg += 1 / np.log2(rank + 2)
            ideal_dcg = 0.0
            for rank in range(min(len(true_items), 10)):
                ideal_dcg += 1 / np.log2(rank + 2)
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        final_recall = np.mean(recall_scores)
        final_ndcg   = np.mean(ndcg_scores)
        print(f"Recall@10  : {final_recall:.4f}")
        print(f"NDCG@10    : {final_ndcg:.4f}")

        results.append({
            "name"   : CONFIG["name"],
            "recall" : round(final_recall, 4),
            "ndcg"   : round(final_ndcg,   4),
            "params" : total_params,
        })

    # ── Summary results table ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS TABLE — Recall@10 and NDCG@10")
    print(f"{'=' * 60}")
    print(f"{'Configuration':<42} {'Recall@10':>10} {'NDCG@10':>10}")
    print(f"{'─' * 42} {'─' * 10} {'─' * 10}")
    for r in results:
        print(f"{r['name']:<42} {r['recall']:>10.4f} {r['ndcg']:>10.4f}")
    print(f"{'=' * 60}")

    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    with open("./results/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved → results/eval_results.json")

    # ── Figure 1: Loss curves (one subplot per model) ─────────────────────────
    curves = []
    for CONFIG in CONFIGS:
        hp = CONFIG.get("history_path", "")
        if hp and os.path.exists(hp):
            with open(hp) as f:
                curves.append((CONFIG["name"], json.load(f)))

    # One colour per model so curves are visually distinct
    curve_colors = ["#2166AC", "#1A9850", "#D6604D"]

    if curves:
        fig, axes = plt.subplots(1, len(curves),
                                 figsize=(5 * len(curves), 4.5),
                                 sharey=False)
        if len(curves) == 1:
            axes = [axes]

        for idx, (ax, (name, hist)) in enumerate(zip(axes, curves)):
            color = curve_colors[idx % len(curve_colors)]
            epochs_ran = list(range(1, len(hist["train_loss"]) + 1))

            # Find best val epoch for vertical marker
            best_ep = int(np.argmin(hist["val_loss"])) + 1

            ax.plot(epochs_ran, hist["train_loss"],
                    color=color, linewidth=2, marker="o", markersize=3,
                    label="Train Loss")
            ax.plot(epochs_ran, hist["val_loss"],
                    color=color, linewidth=2, marker="o", markersize=3,
                    linestyle="--", alpha=0.75, label="Val Loss")

            # Vertical dotted line at best val epoch
            ax.axvline(x=best_ep, color="grey", linestyle=":",
                       linewidth=1.2, alpha=0.8)
            ax.text(best_ep + 0.15,
                    ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.16,
                    "best\nval", fontsize=7, color="grey", va="bottom")

            ax.set_title(name, fontsize=9, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("BCE Loss", fontsize=9)
            ax.set_xlim(left=1)
            ax.set_ylim(bottom=0.15)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)

        fig.suptitle("Training and Validation Loss — NCF Models",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("./results/loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Loss curves saved → results/loss_curves.png")

    # ── Figure 2: Bar charts comparing Recall@10 and NDCG@10 ─────────────────
    if results:
        # Short x-axis labels
        short_names = []
        for r in results:
            n = r["name"]
            if n.startswith("MLP"):
                short_names.append("MLP")
            elif n.startswith("GMF"):
                short_names.append("GMF")
            else:
                short_names.append("NeuMF")

        recalls = [r["recall"] for r in results]
        ndcgs   = [r["ndcg"]   for r in results]
        bar_colors = ["#2166AC", "#1A9850", "#D6604D"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # ── subplot 1: Recall@10 ──────────────────────────────────────────────
        ax = axes[0]
        bars = ax.bar(short_names, recalls,
                      color=bar_colors[:len(results)], width=0.5, zorder=3)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.axhline(y=0.05, color="red", linestyle="--",
                   linewidth=1, label="Min threshold (0.05)")
        ax.set_title("Recall@10 Comparison", fontsize=11, fontweight="bold")
        ax.set_ylabel("Recall@10", fontsize=10)
        ax.set_ylim(0, max(recalls) + 0.04)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

        # ── subplot 2: NDCG@10 ───────────────────────────────────────────────
        ax = axes[1]
        bars = ax.bar(short_names, ndcgs,
                      color=bar_colors[:len(results)], width=0.5, zorder=3)
        for bar, val in zip(bars, ndcgs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.axhline(y=0.03, color="red", linestyle="--",
                   linewidth=1, label="Min threshold (0.03)")
        ax.set_title("NDCG@10 Comparison", fontsize=11, fontweight="bold")
        ax.set_ylabel("NDCG@10", fontsize=10)
        ax.set_ylim(0, max(ndcgs) + 0.06)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

        # ── subplot 3: Recall@10 vs NDCG@10 grouped ──────────────────────────
        ax = axes[2]
        x     = np.arange(len(results))
        width = 0.35

        bars_r = ax.bar(x - width / 2, recalls, width,
                        color=bar_colors[:len(results)],
                        label="Recall@10", zorder=3)
        bars_n = ax.bar(x + width / 2, ndcgs, width,
                        color=bar_colors[:len(results)],
                        alpha=0.45, hatch="///",
                        label="NDCG@10", zorder=3)

        for bar, val in zip(bars_r, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars_n, ndcgs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_title("Recall@10 vs NDCG@10", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0, max(max(recalls), max(ndcgs)) + 0.06)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

        fig.suptitle("NCF Model Comparison — MovieLens 1M",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("./results/bar_charts.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Bar charts saved → results/bar_charts.png")


if __name__ == "__main__":
    evaluate()