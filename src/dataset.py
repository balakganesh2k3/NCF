"""
dataset.py — Part 1: Data Pipeline
NCF Assignment — MovieLens 1M
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load ratings.dat
# ─────────────────────────────────────────────────────────────────────────────

def load_data(file_path):
    print("=" * 50)
    print("STEP 1 — Load & binarize")
    print("=" * 50)
    dataframe = pd.read_csv(
        file_path,
        sep='::',
        engine='python',
        names=['user', 'movie', 'rating', 'time'],
        encoding='latin-1'
    )
    print(f"Total ratings loaded  : {len(dataframe):,}")
    print(f"Unique users          : {dataframe['user'].nunique():,}")
    print(f"Unique movies         : {dataframe['movie'].nunique():,}")
    print(f"Rating distribution   :")
    print(dataframe['rating'].value_counts().sort_index().to_string())
    return dataframe


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Keep only positive interactions (rating >= 4)
# ─────────────────────────────────────────────────────────────────────────────

def get_positive_interactions(dataframe):
    positive = dataframe[dataframe['rating'] >= 4].copy()
    positive['label'] = 1
    positive = positive[['user', 'movie', 'label']].reset_index(drop=True)
    print(f"\nPositive interactions : {len(positive):,}  (rating >= 4)")
    print(f"Dropped               : {len(dataframe) - len(positive):,}  (rating < 4)")
    return positive


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Re-index users and items to 0-based consecutive integers
# ─────────────────────────────────────────────────────────────────────────────

def reindexing(dataframe):
    print("\n" + "=" * 50)
    print("STEP 2 — Re-index")
    print("=" * 50)
    users  = sorted(dataframe['user'].unique())
    movies = sorted(dataframe['movie'].unique())
    index_user  = {u_id: i for i, u_id in enumerate(users)}
    index_movie = {m_id: i for i, m_id in enumerate(movies)}
    dataframe = dataframe.copy()
    dataframe['user']  = dataframe['user'].map(index_user)
    dataframe['movie'] = dataframe['movie'].map(index_movie)
    total_users  = len(users)
    total_movies = len(movies)
    print(f"num_users : {total_users}  (IDs: 0 → {total_users - 1})")
    print(f"num_items : {total_movies}  (IDs: 0 → {total_movies - 1})")
    print(f"user  range : {dataframe['user'].min()} – {dataframe['user'].max()}")
    print(f"movie range : {dataframe['movie'].min()} – {dataframe['movie'].max()}")
    return dataframe, total_users, total_movies


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Split positives into train / val / test (70 / 15 / 15) per user
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(dataframe, seed=42):
    """
    Split per user so every user appears in train.
    Each user's positive items are shuffled then split 70/15/15.
    """
    print("\n" + "=" * 50)
    print("STEP 3 — Train / Val / Test split")
    print("=" * 50)
    rnd    = random.Random(seed)
    groups = dataframe.groupby('user')['movie'].apply(list)
    training_rows, validation_rows, testing_rows = [], [], []

    for u, items in groups.items():
        items = items.copy()
        rnd.shuffle(items)
        n = len(items)
        if n == 1:
            w1, w2, w3 = items, [], []
        elif n == 2:
            w1, w2, w3 = items[:1], items[1:2], []
        else:
            p1 = max(1, int(round(n * 0.70)))
            p2 = max(0, int(round(n * 0.15)))
            p3 = n - p1 - p2
            if p3 == 0 and p2 > 0 and p1 > 1:
                p1 -= 1
                p3 += 1
            w1 = items[:p1]
            w2 = items[p1:p1 + p2]
            w3 = items[p1 + p2:]
        training_rows.extend([(int(u), int(i), 1) for i in w1])
        validation_rows.extend([(int(u), int(i), 1) for i in w2])
        testing_rows.extend([(int(u), int(i), 1) for i in w3])

    train_pos = pd.DataFrame(training_rows,   columns=['user', 'movie', 'label'])
    val_pos   = pd.DataFrame(validation_rows, columns=['user', 'movie', 'label'])
    test_pos  = pd.DataFrame(testing_rows,    columns=['user', 'movie', 'label'])

    total = len(train_pos) + len(val_pos) + len(test_pos)
    print(f"Train : {len(train_pos):,}  ({len(train_pos)/total*100:.1f}%)")
    print(f"Val   : {len(val_pos):,}   ({len(val_pos)/total*100:.1f}%)")
    print(f"Test  : {len(test_pos):,}   ({len(test_pos)/total*100:.1f}%)")
    print(f"Total : {total:,}")

    # Verify zero overlap of positive (user, movie) pairs
    train_pairs = set(zip(train_pos.user, train_pos.movie))
    val_pairs   = set(zip(val_pos.user,   val_pos.movie))
    test_pairs  = set(zip(test_pos.user,  test_pos.movie))
    assert len(train_pairs & val_pairs)  == 0, "Overlap between train and val!"
    assert len(train_pairs & test_pairs) == 0, "Overlap between train and test!"
    assert len(val_pairs   & test_pairs) == 0, "Overlap between val and test!"
    print("\nZero overlap between all splits ✓")

    return train_pos, val_pos, test_pos


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Negative sampling (per user, never global)
# ─────────────────────────────────────────────────────────────────────────────

def get_negativesampling(split_pos, all_pos, total_movies, rat=4, seed=42):
    """
    Sample `rat` negatives per positive for a given split.

    Args:
        split_pos   : positives for this split (train or val)
        all_pos     : ALL positives across all splits — used to ensure
                      sampled negatives are never true positives for the user
        total_movies: total number of re-indexed items
        rat         : negatives per positive (default 4)
        seed        : random seed

    Returns:
        DataFrame of negatives [user, movie, label=0]
    """
    # Build full interaction history per user across ALL splits
    done       = all_pos.groupby('user')['movie'].apply(set).to_dict()
    all_movies = set(range(total_movies))
    output     = []
    counts     = split_pos.groupby('user')['movie'].count().to_dict()
    rnd        = random.Random(seed)

    for u_id, c in counts.items():
        n_required = rat * int(c)
        pool       = list(all_movies - done.get(u_id, set()))
        if not pool:
            continue
        chosen = rnd.sample(pool, min(n_required, len(pool)))
        for m_id in chosen:
            output.append({'user': int(u_id), 'movie': int(m_id), 'label': 0})

    return pd.DataFrame(output, columns=['user', 'movie', 'label'])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Combine positives + negatives and shuffle
# ─────────────────────────────────────────────────────────────────────────────

def add(pos_split, neg_split, seed=42):
    data     = pos_split[['user', 'movie', 'label']].copy()
    neg_data = neg_split[['user', 'movie', 'label']].copy()
    complete = pd.concat([data, neg_data], ignore_index=True)
    complete = complete.drop_duplicates(subset=['user', 'movie'])
    complete = complete.sample(frac=1, random_state=seed).reset_index(drop=True)
    return complete


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MovieDataset(Dataset):
    """
    PyTorch Dataset wrapping (user, movie, label) triples.
    Labels stored as float32 — compatible with BCELoss directly.
    """
    def __init__(self, dataframe):
        self.users  = dataframe['user'].astype(np.int64).values.copy()
        self.movies = dataframe['movie'].astype(np.int64).values.copy()
        self.labels = dataframe['label'].astype(np.float32).values.copy()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx],  dtype=torch.long),
            torch.tensor(self.movies[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Build DataLoaders (main entry point used by train.py + evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(filepath, batch_size=256, neg_ratio=4, num_workers=0, seed=42):
    """
    Run the full pipeline and return DataLoaders ready for model.py.

    Pipeline:
        load_data → get_positive_interactions → reindexing →
        split_dataset → get_negativesampling → add → MovieDataset → DataLoader

    Args:
        filepath    : path to ratings.dat
        batch_size  : batch size for all loaders
        neg_ratio   : negatives per positive in train/val (default 4)
        num_workers : DataLoader workers (0 = main process)
        seed        : random seed

    Returns:
        train_loader  : DataLoader — train positives + negatives (shuffled)
        val_loader    : DataLoader — val positives + negatives
        test_loader   : DataLoader — test positives ONLY (no negatives)
        num_users     : int — pass to model constructor
        num_items     : int — pass to model constructor
        user_history  : dict user -> set of ALL interacted movie IDs
                        (train + val only, used in evaluate.py for masking)
    """
    # Step 1 — Load
    raw = load_data(filepath)

    # Step 2 — Binarize
    positives = get_positive_interactions(raw)

    # Step 3 — Re-index
    positives, num_users, num_items = reindexing(positives)

    # Step 4 — Split positives
    train_pos, val_pos, test_pos = split_dataset(positives, seed=seed)

    # Step 5 — Negative sampling for train and val only
    # Test stays positives-only for full-ranking evaluation (Recall@10, NDCG@10)
    print("\n" + "=" * 50)
    print("STEP 4 — Negative sampling (training only)")
    print("=" * 50)
    neg_train = get_negativesampling(train_pos, positives, num_items, rat=neg_ratio, seed=seed)
    print(f"Positives  : {len(train_pos):,}")
    print(f"Negatives  : {len(neg_train):,}  ({neg_ratio}:1 ratio)")

    print("\n" + "=" * 50)
    print("STEP 4 — Negative sampling (validation)")
    print("=" * 50)
    neg_val = get_negativesampling(val_pos, positives, num_items, rat=neg_ratio, seed=seed + 1)
    print(f"Positives  : {len(val_pos):,}")
    print(f"Negatives  : {len(neg_val):,}  ({neg_ratio}:1 ratio)")

    # Step 6 — Combine
    train_df = add(train_pos, neg_train, seed=seed)
    val_df   = add(val_pos,   neg_val,   seed=seed + 1)
    # test_df is positives only — no negatives added
    test_df  = test_pos[['user', 'movie', 'label']].copy().reset_index(drop=True)

    # Verify labels
    assert set(train_df['label'].unique()).issubset({0.0, 1.0, 0, 1}), "Bad train labels"
    assert set(val_df['label'].unique()).issubset({0.0, 1.0, 0, 1}),   "Bad val labels"
    assert set(test_df['label'].unique()) == {1} or set(test_df['label'].unique()) == {1.0}, \
        "Test set should be positives only"
    print("\nLabel check: only 0 and 1 present ✓")

    # Step 7 — Build user_history (train + val positives only, NOT test)
    # evaluate.py masks seen items using this so test positives stay as candidates
    user_history = defaultdict(set)
    for row in train_pos.itertuples(index=False):
        user_history[row.user].add(row.movie)
    for row in val_pos.itertuples(index=False):
        user_history[row.user].add(row.movie)
    user_history = dict(user_history)

    # Step 8 — Datasets + DataLoaders
    train_dataset = MovieDataset(train_df)
    val_dataset   = MovieDataset(val_df)
    test_dataset  = MovieDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("\n" + "=" * 50)
    print("STEP 5+6 — Datasets and DataLoaders")
    print("=" * 50)
    print(f"train_dataset : {len(train_dataset):,} samples")
    print(f"val_dataset   : {len(val_dataset):,} samples")
    print(f"test_dataset  : {len(test_dataset):,} samples  (positives only)")
    print(f"Batch size    : {batch_size}")
    print(f"Train batches : {len(train_loader):,}")
    print(f"Val batches   : {len(val_loader):,}")
    print(f"Test batches  : {len(test_loader):,}")

    # Acceptance criteria
    print("\n" + "=" * 50)
    print("ACCEPTANCE CRITERIA")
    print("=" * 50)
    u_batch, i_batch, l_batch = next(iter(train_loader))
    print(f"Batch dtypes   : users={u_batch.dtype}  items={i_batch.dtype}  labels={l_batch.dtype}")
    print(f"Batch shapes   : users={u_batch.shape}  items={i_batch.shape}  labels={l_batch.shape}")
    assert u_batch.dtype == torch.int64,   "user IDs must be LongTensor"
    assert i_batch.dtype == torch.int64,   "item IDs must be LongTensor"
    assert l_batch.dtype == torch.float32, "labels must be FloatTensor for BCELoss"
    print("Tensor dtype check ✓")
    print(f"\nAll done. Ready for model.py.")
    print(f"  num_users = {num_users}")
    print(f"  num_items = {num_items}")

    return train_loader, val_loader, test_loader, num_users, num_items, user_history


# ─────────────────────────────────────────────────────────────────────────────
# Run directly: python dataset.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/ml-1m/ratings.dat"
    build_dataloaders(filepath=path, batch_size=256, neg_ratio=4, seed=42)