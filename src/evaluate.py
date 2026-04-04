""" Evaluate the performance using Recall@10, NDCG@10. """

from dataset import build_dataloaders
from model   import gmf_lay, mlp_lay, neumf_lay

import numpy  as np
import torch
import json
import matplotlib.pyplot as plt

CONFIGS = [
    {
    "ratings_path" : "../data/ml-1m/ratings.dat",
    "batch_size" : 256,
    "neg_ratio" : 4,
    "model_type" : "mlp",        
    "num_factors" : 32,           
    "layers" : [64, 32, 16],   
    "lr" : 0.001,
    "epochs" : 20,
    "patience" : 5,
    "checkpoint_path": "checkpoints/mlp_32.pt",
    "history_path" : "results/train_mlp_32.json",
    "seed" : 42,
    },
    {
    "ratings_path" : "../data/ml-1m/ratings.dat",
    "batch_size" : 256,
    "neg_ratio" : 4,
    "model_type" : "gmf",        
    "num_factors" : 32,           
    "layers" : [64, 32, 16],   
    "lr" : 0.001,
    "epochs" : 20,
    "patience" : 5,
    "checkpoint_path": "checkpoints/gmf_32.pt",
    "history_path" : "results/train_gmf_32.json",
    "seed" : 42,
    },
    {
    "ratings_path" : "../data/ml-1m/ratings.dat",
    "batch_size" : 256,
    "neg_ratio" : 4,
    "model_type" : "neumf",        
    "num_factors" : 32,           
    "layers" : [64, 32, 16],   
    "lr" : 0.001,
    "epochs" : 20,
    "patience" : 5,
    "checkpoint_path": "checkpoints/neumf_64_32_16.pt",
    "history_path" : "results/train_neumf_64_32_16.json",
    "seed" : 42,
    },
    {
    "ratings_path" : "../data/ml-1m/ratings.dat",
    "batch_size" : 256,
    "neg_ratio" : 4,
    "model_type" : "neumf",        
    "num_factors" : 32,           
    "layers" : [128, 64, 32],   
    "lr" : 0.001,
    "epochs" : 20,
    "patience" : 5,
    "checkpoint_path": "/checkpoints/neumf_128_64_32.pt",
    "history_path" : "results/train_neumf_128_64_32.json",
    "seed" : 42,
    }
]

def evaluate():
    print("\nLoading data")

    print("\nReconstructing  model")
    for CONFIG in CONFIGS:
        train_loader, val_loader, test_loader, num_users, num_items, user_history = build_dataloaders (filepath = CONFIG["ratings_path"], 
                                                                                                  batch_size = CONFIG["batch_size"], 
                                                                                                  neg_ratio = CONFIG["neg_ratio"], 
                                                                                                  seed = CONFIG["seed"])
        model_type = CONFIG["model_type"]
        if model_type == "neumf":
            model = neumf_lay(num_users = num_users, num_items = num_items, num_factors = CONFIG["num_factors"], layers = CONFIG["layers"])
        elif model_type == "gmf":
            model = gmf_lay(num_users = num_users, num_items = num_items, num_factors = CONFIG["num_factors"])
        elif model_type == "mlp":
            model = mlp_lay(num_users = num_users, num_items = num_items, layers = CONFIG["layers"])
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'neumf', 'gmf', or 'mlp'.")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"model_type : {model_type} | params : {total_params:,}")

# load trained weights into the model and switch to evaluation mode

        model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location="cpu"))
        model.eval()
        print(f"checkpoint loaded from : {CONFIG['checkpoint_path']}")

# for each user collect the set of items that they actually liked in the set

        ground_truth = {}
        for user_ids, item_ids, labels in test_loader:
            for u, i, l in zip(user_ids, item_ids, labels):
                if l.item() == 1:
                    # if the user is not already added create empty set for them
                    if u.item() not in ground_truth:
                        ground_truth[u.item()] = set()
                    # if user is already present add item to set
                    ground_truth[u.item()].add(i.item())
    
        recall_scores = []
        ndcg_scores = []

# create set of movies the user has not seen
        all_items = set(range(num_items))
        for user_id, true_items in ground_truth.items():
            seen_items  = user_history.get(user_id, set())
            candidates  = list(all_items - seen_items)
            userid_tensor = torch.tensor([user_id])
            user_tensor = userid_tensor.repeat(len(candidates))
            candidate_tensor = torch.tensor(candidates)
# get scores     
            with torch.no_grad():
                scores = model(user_tensor, candidate_tensor)        
# sort the scores in descending order and get the top 10 items        
            top_indices = torch.topk(scores, 10, largest = True).indices
            top_items = [candidates[idx] for idx in top_indices]

# find items that user actually liked
            hits = len(set(top_items) & true_items)

# implementing Recall@10
            recall = hits/len(true_items)
            recall_scores.append(recall)

# implementing DCG
            dcg = 0.0
            for rank,item in enumerate(top_items):
                if item in true_items:
                    dcg += 1/ np.log2(rank +2)  
            ideal_dcg = 0.0
            for rank in range(min(len(true_items), 10)):
                    ideal_dcg += 1/ np.log2(rank +2)
                
            ndcg = dcg/ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        final_recall = np.mean(recall_scores)
        final_ndcg = np.mean(ndcg_scores)
        print(f"Recall@10: {final_recall:.4f}")
        print(f"NDCG@10  : {final_ndcg:.4f}")

        with open("../results/train_mlp_32.json") as f:
            mlp = json.load(f)
        with open("../results/train_gmf_32.json") as f:
            gmf = json.load(f)
        with open("../results/train_neumf_64_32_16.json") as f:
            neumf = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(mlp["train_loss"], label="Train Loss")
        axes[0].plot(mlp["val_loss"], label="Val Loss")
        axes[0].set_title("MLP [64, 32, 16]")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BCE Loss")
        axes[0].legend()

        axes[1].plot(gmf["train_loss"], label="Train Loss")
        axes[1].plot(gmf["val_loss"], label="Val Loss")
        axes[1].set_title("GMF (factors=32)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("BCE Loss")
        axes[1].legend()

        axes[2].plot(neumf["train_loss"], label="Train Loss")
        axes[2].plot(neumf["val_loss"], label="Val Loss")
        axes[2].set_title("NeuMF [64, 32, 16]")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("BCE Loss")
        axes[2].legend()

        plt.suptitle("Training and Validation Loss", fontsize=14)
        plt.tight_layout()
        plt.savefig("../results/loss_curves.png", dpi=150)


if __name__ == "__main__":
    evaluate()
