import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
def load_ratings(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep="::",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
    )
    print("=" * 50)
    print("Load & binarize")
    print("=" * 50)
    print(f"Total ratings loaded : {len(df):,}")
    print(f"Unique users : {df['userId'].nunique():,}")
    print(f"Unique movies : {df['movieId'].nunique():,}")
    print(f"Rating distribution :")
    print(df["rating"].value_counts().sort_index().to_string())
    positives = df[df["rating"] >= 4][["userId", "movieId"]].copy()
    positives["label"] = 1
    positives = positives.reset_index(drop=True)
    print(f"\nPositive interactions : {len(positives):,} (rating >= 4)")
    print(f"Dropped : {len(df) - len(positives):,} (rating < 4)")
    return positives
def reindex(df: pd.DataFrame):
    unique_users = sorted(df["userId"].unique())
    unique_items = sorted(df["movieId"].unique())
    user2idx = {u: idx for idx, u in enumerate(unique_users)}
    item2idx = {i: idx for idx, i in enumerate(unique_items)}
    df = df.copy()
    df["userId"] = df["userId"].map(user2idx)
    df["movieId"] = df["movieId"].map(item2idx)
    num_users = len(user2idx)
    num_items = len(item2idx)
    print("\n" + "=" * 50)
    print("Re-index")
    print("=" * 50)
    print(f"num_users : {num_users} (IDs: 0 → {num_users - 1})")
    print(f"num_items : {num_items} (IDs: 0 → {num_items - 1})")
    print(f"userId range : {df['userId'].min()} – {df['userId'].max()}")
    print(f"movieId range : {df['movieId'].min()} – {df['movieId'].max()}")
    return df, user2idx, item2idx, num_users, num_items
def split_data(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, seed=42):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_pos = shuffled.iloc[:n_train].reset_index(drop=True)
    val_pos = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_pos = shuffled.iloc[n_train + n_val :].reset_index(drop=True)
    print("\n" + "=" * 50)
    print("Train / Val / Test split")
    print("=" * 50)
    print(f"Train : {len(train_pos):,} ({len(train_pos)/n*100:.1f}%)")
    print(f"Val : {len(val_pos):,} ({len(val_pos)/n*100:.1f}%)")
    print(f"Test : {len(test_pos):,} ({len(test_pos)/n*100:.1f}%)")
    print(f"Total : {len(train_pos) + len(val_pos) + len(test_pos):,}")
    train_pairs = set(zip(train_pos.userId, train_pos.movieId))
    val_pairs = set(zip(val_pos.userId, val_pos.movieId))
    test_pairs = set(zip(test_pos.userId, test_pos.movieId))
    assert len(train_pairs & val_pairs) == 0, "Overlap between train and val!"
    assert len(train_pairs & test_pairs) == 0, "Overlap between train and test!"
    assert len(val_pairs & test_pairs) == 0, "Overlap between val and test!"
    print("\nZero overlap between all splits ✓")
    return train_pos, val_pos, test_pos
def negative_sampling(
    train_pos : pd.DataFrame,
    all_pos : pd.DataFrame,
    num_items : int,
    neg_ratio : int = 4,
    seed : int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    user_history = (
        all_pos.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )
    all_items = set(range(num_items))
    negatives = []
    for user_id, group in train_pos.groupby("userId"):
        pos_items = user_history.get(user_id, set())
        neg_pool = list(all_items - pos_items)
        n_needed = len(group) * neg_ratio
        n_sample = min(n_needed, len(neg_pool))  
        sampled = random.sample(neg_pool, n_sample)
        for item_id in sampled:
            negatives.append({
                "userId" : user_id,
                "movieId" : item_id,
                "label" : 0,
            })
    neg_df = pd.DataFrame(negatives)
    train_df = pd.concat([train_pos, neg_df], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print("\n" + "=" * 50)
    print("Negative sampling (training only)")
    print("=" * 50)
    print(f"Positives : {len(train_pos):,}")
    print(f"Negatives : {len(neg_df):,}  ({neg_ratio}:1 ratio)")
    print(f"Total : {len(train_df):,}")
    print(f"Label counts:\n{train_df['label'].value_counts().to_string()}")
    assert set(train_df["label"].unique()) == {0, 1}, "Labels must be 0 or 1 only!"
    print("\nLabel check: only 0 and 1 present ✓")
    return train_df
class NCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users = torch.LongTensor(df["userId"].values.copy())
        self.items = torch.LongTensor(df["movieId"].values.copy())
        self.labels = torch.LongTensor(df["label"].values.copy())
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
def build_dataloaders(
    filepath : str,
    batch_size : int = 256,
    neg_ratio : int = 4,
    num_workers : int = 0,
    seed : int = 42,
):
    positives = load_ratings(filepath)
    positives, user2idx, item2idx, num_users, num_items = reindex(positives)
    train_pos, val_pos, test_pos = split_data(positives, seed=seed)
    user_history = positives.groupby("userId")["movieId"].apply(set).to_dict()
    train_df = negative_sampling(
       train_pos, positives, num_items, neg_ratio=neg_ratio, seed=seed
    )
    val_df = negative_sampling(
      val_pos, positives, num_items, neg_ratio=neg_ratio, seed=seed + 1
    )
    train_dataset = NCFDataset(train_df)
    val_dataset = NCFDataset(val_df)
    test_dataset = NCFDataset(test_pos)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print("\n" + "=" * 50)
    print("STEP 5+6 — Datasets and DataLoaders")
    print("=" * 50)
    print(f"train_dataset : {len(train_dataset):,} samples")
    print(f"val_dataset : {len(val_dataset):,} samples")
    print(f"test_dataset : {len(test_dataset):,} samples")
    print(f"Batch size : {batch_size}")
    print(f"Train batches : {len(train_loader):,}")
    print(f"Val batches : {len(val_loader):,}")
    print(f"Test batches : {len(test_loader):,}")
    print("\n" + "=" * 50)
    print("ACCEPTANCE CRITERIA")
    print("=" * 50)
    batch = next(iter(train_loader))
    user_batch, item_batch, label_batch = batch
    print(f"Batch dtypes : users={user_batch.dtype}  "
          f"items={item_batch.dtype} labels={label_batch.dtype}")
    print(f"Batch shapes : users={user_batch.shape}  "
          f"items={item_batch.shape} labels={label_batch.shape}")
    assert user_batch.dtype == torch.int64, "user IDs must be LongTensor"
    assert item_batch.dtype == torch.int64, "item IDs must be LongTensor"
    assert label_batch.dtype == torch.int64, "labels must be LongTensor"
    print("Tensor dtype check ✓")
    print(f"\nAll done. Ready for model.py.")
    print(f"num_users = {num_users}")
    print(f"num_items = {num_items}")
    return (train_loader, val_loader, test_loader,
            num_users, num_items, user_history)
if __name__ == "__main__":
    import sys
    RATINGS_PATH = "../data/ml-1m/ratings.dat"
    if len(sys.argv) > 1:
        RATINGS_PATH = sys.argv[1]
    (train_loader, val_loader, test_loader,
     num_users, num_items, user_history) = build_dataloaders(
        filepath = RATINGS_PATH,
        batch_size = 256,
        neg_ratio = 4,
        seed = 42,
    )
    print("\n" + "=" * 50)
    print("=" * 50)
   