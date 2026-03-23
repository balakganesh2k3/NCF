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
        num_factors : int  = 8,
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
) -> dict:
    """Train GMF / MLP / NeuMF using BCE loss and Adam optimiser"""
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    print(f"Training on: {device}")
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)
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
                print(f"\nearly stopping triggered after {patience} "
                      f"epochs with no improvement.")
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"\ntraining complete. best val loss: {best_val_loss:.4f}")
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
        assert(out >= 0).all() and (out <= 1).all(), f"{name}: output out of [0,1]"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{name}")
        print(f"Parameters : {n_params:,}")
        print(f"Output : shape={list(out.shape)}"
              f"range=[{out.min():.3f}, {out.max():.3f}]")
        print(f"forward : OK")
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
    print("from model import neumf_lay, train_model")
    print("model = neumf_lay(num_users, num_items, num_factors=8, layers=[64,32,16,8])")
    print("history = train_model(model, train_loader, val_loader)")