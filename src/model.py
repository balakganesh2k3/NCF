import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

def nfc(layer_sizes: list) -> nn.Sequential:
    """Build a stack of Linear → ReLU layers from a list of sizes"""
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class gmf_lay(nn.Module):
    """GMF branch as a standalone model (used for ablation in Part 3)"""
    def __init__(self, num_users: int, num_items: int, num_factors: int = 8):
        super(gmf_lay, self).__init__()
        self.user_emb = nn.Embedding(num_users, num_factors)
        self.item_emb = nn.Embedding(num_items, num_factors)
        self.output_layer = nn.Linear(num_factors, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        nn.init.uniform_(self.output_layer.weight)
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        gmf_vector = u * i
        out = self.output_layer(gmf_vector).squeeze(-1)
        return self.sigmoid(out)


class mlp_lay(nn.Module):
    """MLP branch as a standalone model (used for ablation in Part 3)"""
    def __init__(
        self,
        num_users : int,
        num_items : int,
        layers : list = None,
    ):
        super(mlp_lay, self).__init__()
        if layers is None:
            layers = [64, 32, 16, 8]
        emb_dim = layers[0] // 2
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.mlp = nfc(layers)
        self.output_layer = nn.Linear(layers[-1], 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        mlp_input = torch.cat([u, i], dim=-1)
        mlp_output = self.mlp(mlp_input)
        out = self.output_layer(mlp_output).squeeze(-1)
        return self.sigmoid(out)


class neumf_lay(nn.Module):
    """Neural Matrix Factorization — our full NCF model"""
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
        mlp_emb_dim = layers[0] // 2
        self.gmf_user_emb = nn.Embedding(num_users, num_factors)
        self.gmf_item_emb = nn.Embedding(num_items, num_factors)
        self.mlp_user_emb = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_emb_dim)
        self.mlp = nfc(layers)
        fusion_dim = num_factors + layers[-1]
        self.neuMF_layer = nn.Linear(fusion_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
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
        """
        Initialise NeuMF with weights from pre-trained GMF and MLP models.

        Exactly follows He et al. (2017) Section 3.4:
          - GMF embeddings  → NeuMF GMF embeddings
          - MLP embeddings  → NeuMF MLP embeddings
          - MLP hidden layers → NeuMF MLP hidden layers
          - Fusion output layer weights are the concatenation of the two
            pre-trained output vectors, scaled by alpha (GMF) and
            (1-alpha) (MLP).  Default alpha=0.5 gives equal weighting.

        Args:
            gmf_model : trained gmf_lay instance
            mlp_model : trained mlp_lay instance
            alpha     : weight given to GMF output in fused layer (default 0.5)
        """
        # --- GMF branch embeddings ---
        self.gmf_user_emb.weight.data.copy_(gmf_model.user_emb.weight.data)
        self.gmf_item_emb.weight.data.copy_(gmf_model.item_emb.weight.data)

        # --- MLP branch embeddings ---
        self.mlp_user_emb.weight.data.copy_(mlp_model.user_emb.weight.data)
        self.mlp_item_emb.weight.data.copy_(mlp_model.item_emb.weight.data)

        # --- MLP hidden layers ---
        # nfc() produces [Linear, ReLU, Linear, ReLU, ...] so we copy only
        # the Linear layers, matching them by order.
        mlp_linears_src = [l for l in mlp_model.mlp if isinstance(l, nn.Linear)]
        mlp_linears_dst = [l for l in self.mlp       if isinstance(l, nn.Linear)]
        for src, dst in zip(mlp_linears_src, mlp_linears_dst):
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)

        # --- Fusion output layer (concatenated & scaled) ---
        # gmf output weight shape : (1, num_factors)
        # mlp output weight shape : (1, layers[-1])
        # fused weight shape      : (1, num_factors + layers[-1])
        gmf_out = gmf_model.output_layer.weight.data      # (1, num_factors)
        mlp_out = mlp_model.output_layer.weight.data      # (1, layers[-1])
        fused = torch.cat([alpha * gmf_out, (1 - alpha) * mlp_out], dim=1)
        self.neuMF_layer.weight.data.copy_(fused)

        print(f"Pre-trained weights loaded into NeuMF (alpha={alpha})")

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i
        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp(mlp_in)
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        score = self.neuMF_layer(fused).squeeze(-1)
        return self.sigmoid(score)


def train_model(
    model : nn.Module,
    train_loader : DataLoader,
    val_loader : DataLoader,
    epochs : int = 20,
    lr : float = 0.001,
    patience : int = 5,
    checkpoint_path : str = "best_model.pt",
    device : str = None,
    weight_decay : float = 0.0,
) -> dict:
    """Train GMF / MLP / NeuMF using BCE loss and Adam optimiser.

    Args:
        weight_decay : L2 regularisation coefficient passed to Adam.
                       Keep 0.0 for GMF/MLP from scratch.
                       Use 1e-5 or 1e-6 when fine-tuning pre-trained NeuMF
                       to prevent the pre-trained embeddings from drifting.
    """
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    print(f"Training on: {device}")
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for user_ids, item_ids, labels in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            preds = model(user_ids, item_ids)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
        avg_train_loss = running_loss / len(train_loader.dataset)
        model.eval()
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
                print(f"\nEarly stopping triggered after {patience} "
                      f"epochs with no improvement.")
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return history


if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    print("=" * 55)
    print("check — model.py")
    print("=" * 55)
    NUM_USERS = 6040
    NUM_ITEMS = 3706
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
        print(f"Parameters : {n_params:,}")
        print(f"Output : shape={list(out.shape)}"
              f"range=[{out.min():.3f}, {out.max():.3f}]")
        print(f"forward : OK")

    print("\n--- pretrain weight loading check ---")
    gmf_m = gmf_lay(NUM_USERS, NUM_ITEMS, num_factors=NUM_FACTORS)
    mlp_m = mlp_lay(NUM_USERS, NUM_ITEMS, layers=LAYERS)
    neu_m = neumf_lay(NUM_USERS, NUM_ITEMS, num_factors=NUM_FACTORS, layers=LAYERS)
    neu_m.load_pretrained_weights(gmf_m, mlp_m, alpha=0.5)
    assert torch.allclose(
        neu_m.gmf_user_emb.weight.data, gmf_m.user_emb.weight.data
    ), "GMF user emb mismatch"
    assert torch.allclose(
        neu_m.mlp_user_emb.weight.data, mlp_m.user_emb.weight.data
    ), "MLP user emb mismatch"
    print("Pretrain weight loading: OK")

    print("\n--- 3-epoch training run on NeuMF ---")
    def make_loader(n, with_negatives=False):
        u = torch.randint(0, NUM_USERS, (n,))
        i = torch.randint(0, NUM_ITEMS, (n,))
        if with_negatives:
            pos = n // 5
            l = torch.cat([torch.ones(pos), torch.zeros(n - pos)])[torch.randperm(n)]
        else:
            l = torch.ones(n)
        return DataLoader(TensorDataset(u, i, l.long()), batch_size=BATCH_SIZE, shuffle=True)
    model = neumf_lay(NUM_USERS, NUM_ITEMS, num_factors=NUM_FACTORS, layers=LAYERS)
    train_loader = make_loader(5000, with_negatives=True)
    val_loader   = make_loader(1000, with_negatives=False)
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
    print("\nAll checks passed")
    print("\nUsage in your main script:")
    print("from model import neumf_lay, gmf_lay, mlp_lay, train_model")
    print("gmf = gmf_lay(num_users, num_items, num_factors=32)")
    print("mlp = mlp_lay(num_users, num_items, layers=[64,32,16])")
    print("# train gmf and mlp first, then:")
    print("neumf = neumf_lay(num_users, num_items, num_factors=32, layers=[64,32,16])")
    print("neumf.load_pretrained_weights(gmf, mlp, alpha=0.5)")
    print("history = train_model(neumf, train_loader, val_loader)")