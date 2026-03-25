import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dataset import build_dataloaders
from model import train_model, gmf_lay, mlp_lay, neumf_lay
CONFIG = {
    "ratings" : "./data/ml-1m/ratings.dat",
    "batch_size" : 256,
    "neg_ratio" : 4,
    "num_factors" : 32,
    "layers" : [64, 32, 16],
    "lr" : 0.001,
    "epochs" : 20,
    "patience" : 5,
    "checkpoint_path" : "checkpoints/mlp_lay.pt",
    "history_path" : "results/train_mlp.json",
    "seed" : 42,
}
def main():
    print("=" * 55)
    print("NCF Training — MovieLens 1M")
    print("=" * 55)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print("\nLoading data...")
    (train_loader, val_loader, test_loader,
     num_users, num_items, user_history) = build_dataloaders(
        filepath = CONFIG["ratings"],
        batch_size = CONFIG["batch_size"],
        neg_ratio = CONFIG["neg_ratio"],
        seed = CONFIG["seed"],
    )
    print("\model loading...")
    model = mlp_lay(
     num_users = num_users,
     num_items = num_items,
     layers = [64, 32, 16],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f" num_users : {num_users}")
    print(f" num_items : {num_items}")
   # print(f"  num_factors : {CONFIG['num_factors']}")
    print(f" layers : {CONFIG['layers']}")
    print(f" Parameters : {total_params:,}")
    print("\nStarting training...")
    history = train_model(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        epochs = CONFIG["epochs"],
        lr = CONFIG["lr"],
        patience = CONFIG["patience"],
        checkpoint_path = CONFIG["checkpoint_path"],
    )
    with open(CONFIG["history_path"], "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nLoss history saved → {CONFIG['history_path']}")
    print("\n" + "=" * 55)
    print("TRAINING SUMMARY")
    print("=" * 55)
    print(f"Epochs run : {len(history['train_loss'])}")
    print(f"Best train loss : {min(history['train_loss']):.4f}")
    print(f"Best val loss : {min(history['val_loss']):.4f}")
    print(f"Checkpoint saved : {CONFIG['checkpoint_path']}")
    print(f"History saved : {CONFIG['history_path']}")
if __name__ == "__main__":
    main()