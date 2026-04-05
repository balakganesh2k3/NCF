# Neural collaborative filtering (NCF) dataset preprocessing 
# This file loads the data of the movie ratings, processes it, and creates train/val/test dataloaders
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import sys
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def load_data(file_path):
    print("=" * 50)
    print("step 1 loading data")
    print("=" * 50)
# reading movielens format - user::movie::rating::timestamp
    dataframe = pd.read_csv(file_path, sep='::', engine='python', names=['user', 'movie', 'rating', 'time'], encoding='latin-1')
# displaying dataset stats
    print(f"total ratings loaded : {len(dataframe):,}")
    print(f"unique users : {dataframe['user'].nunique():,}")
    print(f"unique movies : {dataframe['movie'].nunique():,}")
    print(f"rating distribution :")
    print(dataframe['rating'].value_counts().sort_index().to_string())
    return dataframe

def get_positive(dataframe):
    positive = dataframe[dataframe['rating'] >= 4].copy()
    positive['label'] = 1  # marking all the interactions as positive
    positive = positive[['user', 'movie', 'label']].reset_index(drop=True)
    print(f"\npositive interactions : {len(positive):,} (rating >= 4)")
    print(f"dropped : {len(dataframe) - len(positive):,} (rating < 4)")
    return positive
#reindexing the user and movie IDs to continuous integers starting from 0
def reindexing(dataframe):
    print("\n" + "=" * 50)
    print("step 2 reindexing")
    print("=" * 50)
# get unique IDs and sort them
    users = sorted(dataframe['user'].unique())
    movies = sorted(dataframe['movie'].unique())
# Create mapping dictionaries- original_id to new_index
    index_user = {u_id: i for i, u_id in enumerate(users)}
    index_movie = {m_id: i for i, m_id in enumerate(movies)}
# apply mappings to create continuous indices
    dataframe = dataframe.copy()
    dataframe['user'] = dataframe['user'].map(index_user)
    dataframe['movie'] = dataframe['movie'].map(index_movie)
    total_users = len(users)
    total_movies = len(movies)
    print(f"num_users : {total_users} (IDs: 0 → {total_users - 1})")
    print(f"num_items : {total_movies} (IDs: 0 → {total_movies - 1})")
    print(f"user range : {dataframe['user'].min()} – {dataframe['user'].max()}")
    print(f"movie range : {dataframe['movie'].min()} – {dataframe['movie'].max()}")
    return dataframe, total_users, total_movies
#split data per user into train,val and test with approx 70,15 and 15 ratio
#interactions are shuffled to prevent data leakage 
def split_data(dataframe, seed=42):
    print("\n" + "=" * 50)
    print("step 3 train/val/testsplit")
    print("=" * 50)
    rnd = random.Random(seed)
# group all movies by user
    groups = dataframe.groupby('user')['movie'].apply(list)
    train_rows, val_rows, testing_rows = [], [], []
    for u, items in groups.items():
        items = items.copy()
        rnd.shuffle(items)  # randomize the order of user's interactions
        n = len(items)
# handle edge cases where users with 1 or 2 interactions
        if n == 1:
            w1, w2, w3 = items, [], []  # All to train
        elif n == 2:
            w1, w2, w3 = items[:1], items[1:2], []  # split train/val
        else:
# Normal case: aim for 70/15/15 split
            p1 = max(1, int(round(n * 0.70)))
            p2 = max(0, int(round(n * 0.15)))
            p3 = n - p1 - p2
# ensure test set has at least 1 item if possible
            if p3 == 0 and p2 > 0 and p1 > 1:
                p1 -= 1
                p3 += 1
            w1 = items[:p1]
            w2 = items[p1:p1 + p2]
            w3 = items[p1 + p2:]
# Create user, movie and label tuples for each split
        train_rows.extend([(int(u), int(i), 1) for i in w1])
        val_rows.extend([(int(u), int(i), 1) for i in w2])
        testing_rows.extend([(int(u), int(i), 1) for i in w3])
# Convert to dataframes
    train_pos = pd.DataFrame(train_rows, columns=['user', 'movie', 'label'])
    val_pos = pd.DataFrame(val_rows, columns=['user', 'movie', 'label'])
    test_pos = pd.DataFrame(testing_rows, columns=['user', 'movie', 'label'])
# display split stats
    total = len(train_pos) + len(val_pos) + len(test_pos)
    print(f"train : {len(train_pos):,} ({len(train_pos)/total*100:.1f}%)")
    print(f"val : {len(val_pos):,} ({len(val_pos)/total*100:.1f}%)")
    print(f"test : {len(test_pos):,} ({len(test_pos)/total*100:.1f}%)")
    print(f"total : {total:,}")
# Verify no overlap between splits 
    train_pairs = set(zip(train_pos.user, train_pos.movie))
    val_pairs = set(zip(val_pos.user, val_pos.movie))
    test_pairs = set(zip(test_pos.user, test_pos.movie))
    assert len(train_pairs & val_pairs) == 0, "overlap between train and val"
    assert len(train_pairs & test_pairs) == 0, "overlap between train and test"
    assert len(val_pairs & test_pairs) == 0, "overlap between val and test"
    print("\nzero overlap between all splits")
    return train_pos, val_pos, test_pos
#generate negative samples 
def get_negsample(split_pos, all_pos, total_movies, rat=4, seed=42):
    # Build set of movies each user has already interacted with
    done = all_pos.groupby('user')['movie'].apply(set).to_dict()
    all_movies = set(range(total_movies))
    output = []
# count interactions per user in this split
    counts = split_pos.groupby('user')['movie'].count().to_dict()
    rnd = random.Random(seed)
    for u_id, c in counts.items():
# generate negative samples at specified ratio 
        n_required = rat * int(c)
# collection of movies the user hasn't seen 
        pool = list(all_movies - done.get(u_id, set()))
        if not pool:
            continue
# randomly sample from unseen movies
        chosen = rnd.sample(pool, min(n_required, len(pool)))
        for m_id in chosen:
            output.append({'user': int(u_id), 'movie': int(m_id), 'label': 0})
    
    return pd.DataFrame(output, columns=['user', 'movie', 'label'])
def add(pos_split, neg_split, seed=42):
    data = pos_split[['user', 'movie', 'label']].copy()
    neg_data = neg_split[['user', 'movie', 'label']].copy()
# Concatenate positive and negative samples
    complete = pd.concat([data, neg_data], ignore_index=True)
# remove any duplicate (user, movie) pairs
    complete = complete.drop_duplicates(subset=['user', 'movie'])
    
# shuffle randomly to mix positives and negatives
    complete = complete.sample(frac=1, random_state=seed).reset_index(drop=True)
    return complete
class moviedata(Dataset):
    def __init__(self, dataframe):
# convert to numpy arrays for efficient indexing
        self.users = dataframe['user'].astype(np.int64).values.copy()
        self.movies = dataframe['movie'].astype(np.int64).values.copy()
        self.labels = dataframe['label'].astype(np.float32).values.copy()  
    def __len__(self):
        return len(self.users)    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long), # for embedding lookup
            torch.tensor(self.movies[idx], dtype=torch.long), 
            torch.tensor(self.labels[idx], dtype=torch.float32), # For BCELoss
        )
def build_data(filepath, batch_size=256, neg_ratio=4, num_workers=0, seed=42):
# load raw ratings
    raw = load_data(filepath)
    positives = get_positive(raw)
# reindex to continuous IDs
    positives, num_users, num_items = reindexing(positives)
# split into train/val/test
    train_pos, val_pos, test_pos = split_data(positives, seed=seed)
#generate negative samples for training
    print("\n" + "=" * 50)
    print("step 4 negative sampling-training only")
    print("=" * 50)
    neg_train = get_negsample(train_pos, positives, num_items, rat=neg_ratio, seed=seed)
    print(f"positives : {len(train_pos):,}")
    print(f"negatives : {len(neg_train):,} ({neg_ratio}:1 ratio)")
# generate negative samples for validation
    print("\n" + "=" * 50)
    print("step 4 negative sampling-validation")
    print("=" * 50)
    neg_val = get_negsample(val_pos, positives, num_items, rat=neg_ratio, seed=seed + 1)
    print(f"positives : {len(val_pos):,}")
    print(f"negatives : {len(neg_val):,} ({neg_ratio}:1 ratio)")
# combine positive and negative samples
    train_df = add(train_pos, neg_train, seed=seed)
    val_df = add(val_pos, neg_val, seed=seed + 1)
    test_df = test_pos[['user', 'movie', 'label']].copy().reset_index(drop=True) #test has only positives
# validate labels are binary
    assert set(train_df['label'].unique()).issubset({0.0, 1.0, 0, 1}), "bad train labels"
    assert set(val_df['label'].unique()).issubset({0.0, 1.0, 0, 1}), "bad val labels"
    assert set(test_df['label'].unique()) == {1} or set(test_df['label'].unique()) == {1.0}, "test set should be positives"
    print("\nlabel check - only 0 and 1 present")
# build user historytr so we can track all items each user
    user_history = defaultdict(set)
    for row in train_pos.itertuples(index=False):
        user_history[row.user].add(row.movie)
    for row in val_pos.itertuples(index=False):
        user_history[row.user].add(row.movie)
    user_history = dict(user_history)
# create pytorch datasets
    train_dataset = moviedata(train_df)
    val_dataset = moviedata(val_df)
    test_dataset = moviedata(test_df)
# create dataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# display final statistics
    print("\n" + "=" * 50)
    print("step 5+6 datasets and dataLoaders")
    print("=" * 50)
    print(f"train_data : {len(train_dataset):,} samples")
    print(f"val_data : {len(val_dataset):,} samples")
    print(f"test_data : {len(test_dataset):,} samples (positives only)")
    print(f"batch size : {batch_size}")
    print(f"train batches : {len(train_loader):,}")
    print(f"val batches : {len(val_loader):,}")
    print(f"test batches : {len(test_loader):,}")
# validate tensor dtypes are correct for the model
    print("\n" + "=" * 50)
    print("Acceptance criteria")
    print("=" * 50)
    u_batch, i_batch, l_batch = next(iter(train_loader))
    print(f"batch dtypes : users={u_batch.dtype} items={i_batch.dtype} labels={l_batch.dtype}")
    print(f"batch shapes : users={u_batch.shape} items={i_batch.shape} labels={l_batch.shape}")
    assert u_batch.dtype == torch.int64, "user Ids must be longtensor"
    assert i_batch.dtype == torch.int64, "item Ids must be longtensor"
    assert l_batch.dtype == torch.float32, "labels must be floattensor for BCELoss"
    print("tensor dtype check passed")
    print(f"num_users = {num_users}")
    print(f"num_items = {num_items}")
    return train_loader, val_loader, test_loader, num_users, num_items, user_history

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/ml-1m/ratings.dat"
    build_data(filepath=path, batch_size=256, neg_ratio=4, seed=42)