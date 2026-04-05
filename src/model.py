import os 
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

def nfc(layer_sizes: list) -> nn.Sequential:
    layers = []  # will be unpacked into Sequential
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class gmf_lay(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_factors: int = 8):
        super(gmf_lay, self).__init__()
        self.user_emb = nn.Embedding(num_users, num_factors)
        self.item_emb = nn.Embedding(num_items, num_factors)
        self.output_layer = nn.Linear(num_factors, 1, bias=False)  # no bias — He et al. 3.2
        self.sigmoid = nn.Sigmoid()  # squash score to (0,1)
        self.init_weights()  # call after all layers are defined
    def init_weights(self):
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        nn.init.uniform_(self.output_layer.weight)
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)  # (B, num_factors)
        i = self.item_emb(item_ids)  # (B, num_factors)
        gmf_vector = u * i  # element-wise product
        out = self.output_layer(gmf_vector).squeeze(-1) 
        return self.sigmoid(out)  # predicted probability

class mlp_lay(nn.Module):
    def __init__(
        self,
        num_users : int,
        num_items : int,
        layers : list = None,
    ):
        super(mlp_lay, self).__init__()
        if layers is None:
            layers = [64, 32, 16, 8]
        emb_dim = layers[0] // 2  # first layer = concat of two embs, so each gets half
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.mlp = nfc(layers)
        self.output_layer = nn.Linear(layers[-1], 1, bias=False)
        self.sigmoid = nn.Sigmoid()  # squash to (0,1)
        self.init_weights()  # call after all layers are defined
    def init_weights(self):
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)  # (B, emb_dim)
        i = self.item_emb(item_ids)  # (B, emb_dim)
        mlp_input = torch.cat([u, i], dim=-1)  # (B, layers[0])
        mlp_output = self.mlp(mlp_input)  # (B, layers[-1])
        out = self.output_layer(mlp_output).squeeze(-1)  # (B,)
        return self.sigmoid(out)  # predicted probability


class neumf_lay(nn.Module):
    def __init__(
        self,
        num_users : int,
        num_items : int,
        num_factors : int = 8,
        layers : list = None,
    ):
        super(neumf_lay, self).__init__()
        if layers is None:
            layers = [64, 32, 16, 8]
        mlp_emb_dim = layers[0] // 2  # same half-of-first-layer rule as mlp_lay
        self.gmf_user_emb = nn.Embedding(num_users, num_factors)
        self.gmf_item_emb = nn.Embedding(num_items, num_factors)
        self.mlp_user_emb = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_emb_dim)
        self.mlp = nfc(layers)
        fusion_dim = num_factors + layers[-1]  # GMF dims + MLP output dims
        self.neuMF_layer = nn.Linear(fusion_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.neuMF_layer.weight)

    def load_pretrained_weights(
        self,
        gmf_model: gmf_lay,
        mlp_model: mlp_lay,
        alpha: float = 0.5,
    ):
        self.gmf_user_emb.weight.data.copy_(gmf_model.user_emb.weight.data)
        self.gmf_item_emb.weight.data.copy_(gmf_model.item_emb.weight.data)
        self.mlp_user_emb.weight.data.copy_(mlp_model.user_emb.weight.data)
        self.mlp_item_emb.weight.data.copy_(mlp_model.item_emb.weight.data)
        mlp_linears_src = [l for l in mlp_model.mlp if isinstance(l, nn.Linear)]
        mlp_linears_dst = [l for l in self.mlp if isinstance(l, nn.Linear)]
        for src, dst in zip(mlp_linears_src, mlp_linears_dst):
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)
        gmf_out = gmf_model.output_layer.weight.data      # (1, num_factors)
        mlp_out = mlp_model.output_layer.weight.data  # (1, layers[-1])
        fused = torch.cat([alpha * gmf_out, (1 - alpha) * mlp_out], dim=1)  # (1, fusion_dim)
        self.neuMF_layer.weight.data.copy_(fused)
        print(f"pretrained weights loaded into nuemf (alpha={alpha})")

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf_u = self.gmf_user_emb(user_ids)  # (B, num_factors)
        gmf_i = self.gmf_item_emb(item_ids)  # (B, num_factors)
        gmf_out = gmf_u * gmf_i  # (B, num_factors)
        mlp_u = self.mlp_user_emb(user_ids)  # (B, mlp_emb_dim)
        mlp_i = self.mlp_item_emb(item_ids)  # (B, mlp_emb_dim)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=-1)  # (B, layers[0])
        mlp_out = self.mlp(mlp_in)  # (B, layers[-1])
        fused = torch.cat([gmf_out, mlp_out], dim=-1)  # (B, fusion_dim)
        score = self.neuMF_layer(fused).squeeze(-1)  # (B,)
        return self.sigmoid(score)  # predicted probability
    
def train_model(model : nn.Module, train_loader : DataLoader, val_loader : DataLoader, epochs : int = 20, lr : float = 0.001, patience : int = 5, checkpoint_path : str = "best_model.pt", device : str = None, weight_decay : float = 0.0,
) -> dict:
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
    print(f"training on: {device}")
    model = model.to(device)  # move params before building optimizer
    criterion = nn.BCELoss()  # labels and preds both in (0,1)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": []}  # returned for plotting
    best_val_loss = float("inf")
    epochs_no_improve = 0  # early-stopping counter
    for epoch in range(1, epochs + 1):
        model.train()  # enables dropout / batchnorm if present
        running_loss = 0.0
        for user_ids, item_ids, labels in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels = labels.float().to(device)  # BCELoss needs float, not long
            optimizer.zero_grad()
            preds = model(user_ids, item_ids)  # (B,) in [0,1]
            loss = criterion(preds, labels)  # scalar
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)  # weight by batch size
        avg_train_loss = running_loss / len(train_loader.dataset)  # per-sample average
        model.eval()  # disables dropout / batchnorm
        running_val = 0.0
        with torch.no_grad():
            for user_ids, item_ids, labels in val_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                labels = labels.float().to(device)
                preds = model(user_ids, item_ids)
                running_val += criterion(preds, labels).item() * len(labels)
        avg_val_loss = running_val / len(val_loader.dataset)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best — checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nearly stopping triggered after {patience} "
                      f"epochs with no improvement.")
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"\nttraining complete. best val loss: {best_val_loss:.4f}")
    return history

if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    print("=" * 55)
    print("check — model.py")
    print("=" * 55)
    NUM_USERS = 6040  # MovieLens 1M
    NUM_ITEMS = 3706  # MovieLens 1M
    NUM_FACTORS = 8
    LAYERS = [64, 32, 16, 8]
    BATCH_SIZE = 256
    dummy_u = torch.randint(0, NUM_USERS, (BATCH_SIZE,))
    dummy_i = torch.randint(0, NUM_ITEMS, (BATCH_SIZE,))
    for ModelClass, kwargs, name in [
        (gmf_lay, {"num_factors": NUM_FACTORS}, "gmf_lay"),
        (mlp_lay, {"layers": LAYERS}, "mlp_lay"),
        (neumf_lay, {"num_factors": NUM_FACTORS, "layers": LAYERS}, "neumf_lay"),
    ]:
        model = ModelClass(NUM_USERS, NUM_ITEMS, **kwargs)
        with torch.no_grad():
            out = model(dummy_u, dummy_i)
        assert out.shape == (BATCH_SIZE,), f"{name}: wrong output shape"
        assert (out >= 0).all() and (out <= 1).all(), f"{name}: output out of [0,1]"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{name}")
        print(f"parameters : {n_params:,}")
        print(f"output : shape={list(out.shape)}"
              f"range=[{out.min():.3f}, {out.max():.3f}]")
        print(f"forward : OK")
    print("\npretrain weight loading check")
    gmf_m = gmf_lay(NUM_USERS, NUM_ITEMS, num_factors=NUM_FACTORS)
    mlp_m = mlp_lay(NUM_USERS, NUM_ITEMS, layers=LAYERS)
    neu_m = neumf_lay(NUM_USERS, NUM_ITEMS, num_factors=NUM_FACTORS, layers=LAYERS)
    neu_m.load_pretrained_weights(gmf_m, mlp_m, alpha=0.5)
    assert torch.allclose(neu_m.gmf_user_emb.weight.data, gmf_m.user_emb.weight.data), "GMF user emb mismatch"
    assert torch.allclose(neu_m.mlp_user_emb.weight.data, mlp_m.user_emb.weight.data), "MLP user emb mismatch"
    print("pretrain weight loading-OK")
    print("\n3 epoch training run on nuemf")
    def make_loader(n, with_negatives=False):
        u = torch.randint(0, NUM_USERS, (n,))
        i = torch.randint(0, NUM_ITEMS, (n,))
        if with_negatives:
            pos = n // 5  # 4:1 negative ratio
            l = torch.cat([torch.ones(pos), torch.zeros(n - pos)])[torch.randperm(n)]
        else:
            l = torch.ones(n)  # val set is all positives
        return DataLoader(TensorDataset(u, i, l.long()), batch_size=BATCH_SIZE, shuffle=True)
    model = neumf_lay(NUM_USERS, NUM_ITEMS, num_factors=NUM_FACTORS, layers=LAYERS)
    train_loader = make_loader(5000, with_negatives=True)
    val_loader = make_loader(1000, with_negatives=False)
    history = train_model(model, train_loader, val_loader, epochs=3, lr=0.001, patience=3, checkpoint_path="test_ckpt.pt")
    reloaded = neumf_lay(NUM_USERS, NUM_ITEMS, num_factors=NUM_FACTORS, layers=LAYERS)
    reloaded.load_state_dict(torch.load("test_ckpt.pt", map_location="cpu"))
    reloaded.eval()
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(dummy_u, dummy_i), reloaded(dummy_u, dummy_i)), "Checkpoint mismatch!"
    print("Checkpoint reload: OK")
    if os.path.exists("test_ckpt.pt"):
        os.remove("test_ckpt.pt")
    print("\nall checks passed")
    print("\nusage in your main script:")
    print("from model import neumf_lay, gmf_lay, mlp_lay, train_model")
    print("gmf = gmf_lay(num_users, num_items, num_factors=32)")
    print("mlp = mlp_lay(num_users, num_items, layers=[64,32,16])")
    print(" train gmf and mlp first, then:")
    print("neumf = neumf_lay(num_users, num_items, num_factors=32, layers=[64,32,16])")
    print("neumf.load_pretrained_weights(gmf, mlp, alpha=0.5)")
    print("history = train_model(neumf, train_loader, val_loader)")