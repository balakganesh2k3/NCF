import os
import sys
import math
import json
from collections import defaultdict

import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dataset import build_dataloaders
from model   import neumf_lay, gmf_lay, mlp_lay
EXPERIMENTS = [
    {
        "name"            : "NeuMF  [64,32,16]  emb=32",
        "model_class"     : "neumf",
        "num_factors"     : 32,
        "layers"          : [64, 32, 16],
        "checkpoint_path" : "checkpoints/best_model.pt",
    },
    {
        "name"            : "NeuMF  [128,64,32] emb=64",
        "model_class"     : "neumf",
        "num_factors"     : 64,
        "layers"          : [128, 64, 32],
        "checkpoint_path" : "checkpoints/128_64_32.pt",
    },
    {
        "name"            : "GMF-only            emb=32",
        "model_class"     : "gmf",
        "num_factors"     : 32,
        "layers"          : None,
        "checkpoint_path" : "checkpoints/gmf_lay.pt",
    },
    {
        "name"            : "MLP-only  [64,32,16]",
        "model_class"     : "mlp",
        "num_factors"     : None,
        "layers"          : [64, 32, 16],
        "checkpoint_path" : "checkpoints/mlp_lay.pt",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    Fraction of relevant items that appear in the top-k recommended list.

    recall@k = |recommended[:k] ∩ relevant| / |relevant|

    Args:
        recommended : ranked list of item IDs (descending score order)
        relevant    : set of ground-truth positive item IDs for this user
        k           : cutoff (default 10)

    Returns:
        float in [0, 1]
    """
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    hits  = len(top_k & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain at k.

    DCG@k  = sum_i [ rel_i / log2(i + 2) ]   (i is 0-indexed rank)
    IDCG@k = DCG of the ideal ranking (all relevant items at the top)
    NDCG@k = DCG@k / IDCG@k

    Args:
        recommended : ranked list of item IDs (descending score order)
        relevant    : set of ground-truth positive item IDs for this user
        k           : cutoff (default 10)

    Returns:
        float in [0, 1]
    """
    if not relevant:
        return 0.0

    # DCG — reward relevant items found in top-k, discounted by rank
    dcg = 0.0
    for rank, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 2)   # rank+2 because rank is 0-indexed

    # IDCG — best possible DCG (all relevant items at positions 0,1,2,...)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(experiment: dict, num_users: int, num_items: int, device: torch.device):
    """
    Rebuild the model architecture and load saved weights from checkpoint.

    The architecture MUST match exactly what was used during training —
    otherwise load_state_dict will throw a size mismatch error.
    """
    mc = experiment["model_class"]

    if mc == "neumf":
        model = neumf_lay(
            num_users   = num_users,
            num_items   = num_items,
            num_factors = experiment["num_factors"],
            layers      = experiment["layers"],
        )
    elif mc == "gmf":
        model = gmf_lay(
            num_users   = num_users,
            num_items   = num_items,
            num_factors = experiment["num_factors"],
        )
    elif mc == "mlp":
        model = mlp_lay(
            num_users = num_users,
            num_items = num_items,
            layers    = experiment["layers"],
        )
    else:
        raise ValueError(f"Unknown model_class: {mc}")

    checkpoint = experiment["checkpoint_path"]
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"  Make sure you have trained this configuration and saved "
            f"the checkpoint to the correct path."
        )

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()   # disables dropout — critical for reproducible scores
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Full-ranking evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    test_pos        : pd.DataFrame,
    user_history    : dict,
    num_items       : int,
    device          : torch.device,
    k               : int = 10,
    batch_size      : int = 512,
) -> dict:
    """
    Full-ranking evaluation: for each test user, score ALL items not seen
    during training, rank them, and compute Recall@k and NDCG@k.

    Uses matrix (batch) operations — passes all (user, item) pairs for a
    user as a single batch rather than looping item-by-item.

    Args:
        model        : trained model in eval mode
        test_pos     : DataFrame of test positives [userId, movieId]
                       Must contain ONLY positive interactions (label == 1).
        user_history : dict userId -> set of item IDs seen in TRAIN + VAL only.
                       Test positives are NOT included so they remain as
                       valid candidates during ranking.
        num_items    : total number of items
        device       : torch device
        k            : cutoff for Recall and NDCG (default 10)
        batch_size   : how many (user, item) pairs to score at once

    Returns:
        dict with keys: recall, ndcg, n_users
    """
    all_items = torch.arange(num_items, dtype=torch.long)

    # Build per-user test positive set from test_pos
    # test_pos must already be filtered to label == 1 before calling this
    test_pos_per_user = defaultdict(set)
    for row in test_pos.itertuples(index=False):
        test_pos_per_user[row.userId].add(row.movieId)

    test_users = list(test_pos_per_user.keys())

    recalls = []
    ndcgs   = []

    with torch.no_grad():
        for user_id in test_users:
            # FIX 1: user_history here must only contain train+val items,
            # NOT test positives — otherwise the correct answers are masked
            # out and can never appear in the ranked list, giving 0 hits.
            seen_items = user_history.get(user_id, set())

            # Candidate items = all items minus train/val history
            # (test positives are intentionally kept as candidates)
            candidate_mask  = torch.tensor(
                [item not in seen_items for item in range(num_items)],
                dtype=torch.bool
            )
            candidate_items = all_items[candidate_mask]   # shape (n_candidates,)

            if len(candidate_items) == 0:
                continue

            # Score all candidates in mini-batches to avoid OOM on large item sets
            user_tensor = torch.full(
                (len(candidate_items),), user_id, dtype=torch.long, device=device
            )
            scores_list = []
            for start in range(0, len(candidate_items), batch_size):
                end         = start + batch_size
                item_batch  = candidate_items[start:end].to(device)
                user_batch  = user_tensor[start:end]
                score_batch = model(user_batch, item_batch)   # (batch,)
                scores_list.append(score_batch.cpu())

            scores = torch.cat(scores_list)   # (n_candidates,)

            # Sort by descending score and get top-k item IDs
            top_k_indices  = torch.topk(scores, k=min(k, len(scores))).indices
            top_k_items    = candidate_items[top_k_indices].tolist()

            # Ground truth: test positives for this user
            relevant_items = test_pos_per_user[user_id]

            recalls.append(recall_at_k(top_k_items, relevant_items, k))
            ndcgs.append(ndcg_at_k(top_k_items,   relevant_items, k))

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_ndcg   = sum(ndcgs)   / len(ndcgs)   if ndcgs   else 0.0

    return {
        "recall" : mean_recall,
        "ndcg"   : mean_ndcg,
        "n_users": len(recalls),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NCF Evaluation — MovieLens 1M")
    print("=" * 60)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device : {device}")

    # ── Rebuild data pipeline ─────────────────────────────────────────────
    print("\nRebuilding data pipeline...")
    (train_loader, val_loader, test_loader,
     num_users, num_items, user_history) = build_dataloaders(
        filepath   = "./data/ml-1m/ratings.dat",
        batch_size = 256,
        neg_ratio  = 4,
        seed       = 42,
    )
    print(f"num_users = {num_users}, num_items = {num_items}")

    # ── Reconstruct test positives DataFrame ──────────────────────────────
    # FIX 2: Filter to label == 1 only. The test_loader may contain negatives
    # (depending on how build_dataloaders constructs the test set). Including
    # negatives in test_pos_per_user pollutes the ground truth — items with
    # label=0 would be counted as relevant, making recall/NDCG meaningless.
    test_users_list  = []
    test_items_list  = []
    test_labels_list = []
    for u, it, lb in test_loader:
        test_users_list.append(u)
        test_items_list.append(it)
        test_labels_list.append(lb)

    all_test_users  = torch.cat(test_users_list).numpy()
    all_test_items  = torch.cat(test_items_list).numpy()
    all_test_labels = torch.cat(test_labels_list).numpy()

    test_pos_df = pd.DataFrame({
        "userId"  : all_test_users,
        "movieId" : all_test_items,
        "label"   : all_test_labels,
    })
    # Keep only true positives for ground-truth evaluation
    test_pos = test_pos_df[test_pos_df["label"] == 1][["userId", "movieId"]].reset_index(drop=True)
    print(f"Test positives for evaluation : {len(test_pos):,}")

    # ── FIX 3: Build train+val-only history for masking ───────────────────
    # user_history from build_dataloaders contains ALL interactions (train +
    # val + test). If we mask with the full history, the test positives
    # themselves are excluded from the candidate set — the model can never
    # rank them in the top-k, so hits are always 0.
    #
    # We reconstruct a history that contains only train + val items so that
    # test positives remain as valid candidates during full-ranking.
    train_users_list = []
    train_items_list = []
    train_labels_list = []
    for u, it, lb in train_loader:
        train_users_list.append(u)
        train_items_list.append(it)
        train_labels_list.append(lb)

    val_users_list = []
    val_items_list = []
    val_labels_list = []
    for u, it, lb in val_loader:
        val_users_list.append(u)
        val_items_list.append(it)
        val_labels_list.append(lb)

    train_u = torch.cat(train_users_list).numpy()
    train_i = torch.cat(train_items_list).numpy()
    train_l = torch.cat(train_labels_list).numpy()

    val_u = torch.cat(val_users_list).numpy()
    val_i = torch.cat(val_items_list).numpy()
    val_l = torch.cat(val_labels_list).numpy()

    # Build train+val history (positives only — label==1)
    trainval_history: dict = defaultdict(set)
    for u, i, l in zip(train_u, train_i, train_l):
        if l == 1:
            trainval_history[int(u)].add(int(i))
    for u, i, l in zip(val_u, val_i, val_l):
        if l == 1:
            trainval_history[int(u)].add(int(i))

    print(f"Train+val history built for {len(trainval_history):,} users")

    os.makedirs("results", exist_ok=True)

    # ── Run each experiment ───────────────────────────────────────────────
    results = []

    for exp in EXPERIMENTS:
        print(f"\n{'─' * 60}")
        print(f"Evaluating : {exp['name']}")
        print(f"Checkpoint : {exp['checkpoint_path']}")

        try:
            model  = load_model(exp, num_users, num_items, device)
        except FileNotFoundError as e:
            print(f"  SKIPPED — {e}")
            results.append({
                "name"  : exp["name"],
                "recall": "—",
                "ndcg"  : "—",
            })
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters : {n_params:,}")
        print("Scoring all test users (full ranking)...")

        metrics = evaluate(
            model        = model,
            test_pos     = test_pos,
            user_history = trainval_history,   # train+val only — not full history
            num_items    = num_items,
            device       = device,
            k            = 10,
        )

        print(f"  Recall@10 : {metrics['recall']:.4f}")
        print(f"  NDCG@10   : {metrics['ndcg']:.4f}")
        print(f"  Users evaluated : {metrics['n_users']:,}")

        results.append({
            "name"  : exp["name"],
            "recall": round(metrics["recall"], 4),
            "ndcg"  : round(metrics["ndcg"],   4),
        })
    print(f"\n{'=' * 60}")
    print("RESULTS TABLE — Recall@10 and NDCG@10")
    print(f"{'=' * 60}")
    print(f"{'Configuration':<35} {'Recall@10':>10} {'NDCG@10':>10}")
    print(f"{'─' * 35} {'─' * 10} {'─' * 10}")
    for r in results:
        recall_str = f"{r['recall']:.4f}" if isinstance(r["recall"], float) else r["recall"]
        ndcg_str   = f"{r['ndcg']:.4f}"   if isinstance(r["ndcg"],   float) else r["ndcg"]
        print(f"{r['name']:<35} {recall_str:>10} {ndcg_str:>10}")
    print(f"{'=' * 60}")
    results_path = "results/eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")
if __name__ == "__main__":
    main()