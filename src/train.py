import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dataset import build_dataloaders
from model import train_model, gmf_lay, mlp_lay, neumf_lay
CONFIG = {
    "ratings_path" : "./data/ml-1m/ratings.dat",
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
}
def main():
    print("=" * 55)
    print("NCF Training — MovieLens 1M")
    print("=" * 55)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print("\nLoading data")
    (train_loader, val_loader, test_loader, num_users, num_items, user_history) = build_dataloaders(filepath = CONFIG["ratings_path"], batch_size = CONFIG["batch_size"], neg_ratio = CONFIG["neg_ratio"], seed = CONFIG["seed"])
    print("\nbuilding model")
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
    print(f"model_type : {model_type}")
    print(f"num_users : {num_users}")
    print(f"num_items : {num_items}")
    if model_type in ("neumf", "gmf"):
        print(f"num_factors : {CONFIG['num_factors']}")
    if model_type in ("neumf", "mlp"):
        print(f"layers : {CONFIG['layers']}")
    print(f"Parameters : {total_params:,}")
    print("\nStarting training...")
    history = train_model(model = model, train_loader = train_loader, val_loader = val_loader, epochs = CONFIG["epochs"], lr = CONFIG["lr"], patience = CONFIG["patience"], checkpoint_path = CONFIG["checkpoint_path"])
    with open(CONFIG["history_path"], "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nloss history saved → {CONFIG['history_path']}")
    print("\n" + "=" * 55)
    print("Training complete")
    print("=" * 55)
    print(f"model : {model_type}")
    print(f"epochs run : {len(history['train_loss'])}")
    print(f"best train loss : {min(history['train_loss']):.4f}")
    print(f"best val loss : {min(history['val_loss']):.4f}")
    print(f"checkpoint saved : {CONFIG['checkpoint_path']}")
    print(f"history saved : {CONFIG['history_path']}")
if __name__ == "__main__":
    main()