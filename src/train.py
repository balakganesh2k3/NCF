import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))  # make sure local modules are importable
from dataset import build_data
from model import train_model, gmf_lay, mlp_lay, neumf_lay
import torch

Config = {
    "ratings_path" : "./data/ml-1m/ratings.dat",
    "batch_size" : 256,
    "neg_ratio" : 4,  # 4 negatives sampled per positive interaction
    "model_type" : "neumf",  # "neumf" | "gmf" | "mlp"
    "num_factors" : 32,  # embedding size for GMF branch
    "layers" : [64, 32, 16],  # MLP hidden layer widths
    "lr" : 0.0005,
    "epochs" : 20,
    "patience" : 10,  # early-stopping tolerance
    "checkpoint_path" : "checkpoints/neumf_32.pt",
    "history_path" : "results/train_neumf_32.json",
    "seed" : 42,
    "pretrain" : True,  # load GMF+MLP weights before training NeuMF
    "gmf_checkpoint" : "checkpoints/gmf_32.pt",
    "mlp_checkpoint" : "checkpoints/mlp_32.pt",
    "pretrain_alpha" : 0.5,  # blend ratio: alpha*GMF + (1-alpha)*MLP in fusion layer
}

def main():
    print("=" * 55)
    print("NCF training movieLens 1M")
    print("=" * 55)
    os.makedirs("checkpoints", exist_ok=True)  # no-op if already exists
    os.makedirs("results", exist_ok=True)  # same
    print("\nLoading data")
    (train_loader, val_loader, test_loader,
     num_users, num_items, user_history) = build_data(
        filepath = Config["ratings_path"],
        batch_size = Config["batch_size"],
        neg_ratio = Config["neg_ratio"],
        seed = Config["seed"],
    )
    print("\nbuilding model")
    model_type = Config["model_type"]
    if model_type == "neumf":
        model = neumf_lay(
            num_users = num_users,
            num_items = num_items,
            num_factors = Config["num_factors"],
            layers = Config["layers"],
        )
        if Config.get("pretrain", False):
            gmf_ckpt = Config["gmf_checkpoint"]
            mlp_ckpt = Config["mlp_checkpoint"]
            if not os.path.exists(gmf_ckpt):  # fail early with a useful message
                raise FileNotFoundError(
                    f"GMF checkpoint not found: {gmf_ckpt}\n"
                    f"Train GMF first: set model_type='gmf' and run train.py"
                )
            if not os.path.exists(mlp_ckpt):  # same for MLP
                raise FileNotFoundError(
                    f"MLP checkpoint not found: {mlp_ckpt}\n"
                    f"Train MLP first: set model_type='mlp' and run train.py"
                )

            gmf_model = gmf_lay(
                num_users = num_users,
                num_items = num_items,
                num_factors = Config["num_factors"],
            )
            mlp_model = mlp_lay(
                num_users = num_users,
                num_items = num_items,
                layers = Config["layers"],
            )
            gmf_model.load_state_dict(torch.load(gmf_ckpt, map_location="cpu"))  # cpu load — device is set later in train_model
            mlp_model.load_state_dict(torch.load(mlp_ckpt, map_location="cpu"))
            gmf_model.eval()  # freeze batchnorm/dropout before copying weights
            mlp_model.eval()
            print(f"pre-training neumf from:")
            print(f"GMF checkpoint : {gmf_ckpt}")
            print(f"MLP checkpoint : {mlp_ckpt}")
            model.load_pretrained_weights(gmf_model, mlp_model, alpha = Config["pretrain_alpha"],)
        else:
            print("pre-training disabled, training NeuMF from scratch")
    elif model_type == "gmf":
        model = gmf_lay(
            num_users = num_users,
            num_items = num_items,
            num_factors = Config["num_factors"],
        )
    elif model_type == "mlp":
        model = mlp_lay(
            num_users = num_users,
            num_items = num_items,
            layers = Config["layers"],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'neumf', 'gmf', or 'mlp'")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model_type : {model_type}")
    print(f"num_users : {num_users}")
    print(f"num_items : {num_items}")
    if model_type in ("neumf", "gmf"):
        print(f"num_factors : {Config['num_factors']}")
    if model_type in ("neumf", "mlp"):
        print(f"layers : {Config['layers']}")
    print(f"Parameters : {total_params:,}")
    print("\nstarting training")
    history = train_model(model = model, train_loader = train_loader, val_loader = val_loader, epochs = Config["epochs"], lr = Config["lr"], patience = Config["patience"], checkpoint_path = Config["checkpoint_path"])
    with open(Config["history_path"], "w") as f:
        json.dump(history, f, indent=2)  # pretty-print so it's easy to inspect manually
    print(f"\nloss history saved {Config['history_path']}")
    print("\n" + "=" * 55)
    print("Training complete")
    print("=" * 55)
    print(f"model : {model_type}")
    print(f"epochs run : {len(history['train_loss'])}")  # may be less than Config["epochs"] if early stopping fired
    print(f"best train loss : {min(history['train_loss']):.4f}")
    print(f"best val loss : {min(history['val_loss']):.4f}")
    print(f"checkpoint saved : {Config['checkpoint_path']}")
    print(f"history saved : {Config['history_path']}")

if __name__ == "__main__":
    main()