import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
from pathlib import Path

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def load_data(file_path):
    print("Ratings loading in process")
    dataframe = pd.read_csv(
        file_path,
        sep='::',
        engine='python',
        names=['user', 'movie', 'rating', 'time'],
        encoding='latin-1'
    )
    return dataframe

def get_positive_interactions(dataframe):
    positive = dataframe[dataframe['rating'] >= 4].copy()
    positive['label'] = 1
    return positive

def reindexing(dataframe):
    users = sorted(dataframe['user'].unique())
    movies = sorted(dataframe['movie'].unique())
    index_user = {u_id: i for i, u_id in enumerate(users)}
    index_movie = {m_id: i for i, m_id in enumerate(movies)}
    dataframe = dataframe.copy()
    dataframe['user'] = dataframe['user'].map(index_user)
    dataframe['movie'] = dataframe['movie'].map(index_movie)
    total_users = len(users)
    total_movies = len(movies)
    return dataframe, total_users, total_movies

def get_negativesampling(point, takenall, total_movies, rat=4):
    done = takenall.groupby('user')['movie'].apply(set).to_dict()
    all_movies = set(range(total_movies))
    output = []
    counts = point.groupby('user')['movie'].count().to_dict()
    rnd = random.Random(42)
    for u_id, c in counts.items():
        n_required = rat * int(c)
        l = list(all_movies - done.get(u_id, set()))
        if not l:
            continue
        if len(l) >= n_required:
            chosen = rnd.sample(l, n_required)
        else:
            chosen = l
        for m_id in chosen:
            output.append({'user': int(u_id), 'movie': int(m_id), 'label': 0})
    neint = pd.DataFrame(output, columns=['user','movie','label'])
    return neint

def add(point, neint):
    data = point[['user', 'movie', 'label']].copy()
    n_data = neint[['user', 'movie', 'label']].copy()
    complete = pd.concat([data, n_data], ignore_index=True)
    complete = complete.drop_duplicates(subset=['user', 'movie', 'label'])
    complete = complete.sample(frac=1, random_state=42).reset_index(drop=True)
    return complete

def split_dataset(dataframe):
    rnd = random.Random(42)
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
            p3 = max(0, n - p1 - p2)
            if p3 == 0 and p2 > 0 and p1 > 1:
                p1 -= 1
                p3 += 1
            w1 = items[:p1]
            w2 = items[p1:p1+p2]
            w3 = items[p1+p2:p1+p2+p3]
        training_rows.extend([(int(u), int(i), 1) for i in w1])
        validation_rows.extend([(int(u), int(i), 1) for i in w2])
        testing_rows.extend([(int(u), int(i), 1) for i in w3])
    training = pd.DataFrame(training_rows, columns=['user','movie','label'])
    validation = pd.DataFrame(validation_rows, columns=['user','movie','label'])
    testing = pd.DataFrame(testing_rows, columns=['user','movie','label'])
    return training, validation, testing

class MovieDataset(Dataset):
    def __init__(self, dataframe):
        self.users = dataframe['user'].astype(np.int64).values
        self.movies = dataframe['movie'].astype(np.int64).values
        self.labels = dataframe['label'].astype(np.float32).values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, id):
        user = torch.tensor(self.users[id], dtype=torch.long)
        movie = torch.tensor(self.movies[id], dtype=torch.long)
        label = torch.tensor(self.labels[id], dtype=torch.float32)
        return user, movie, label

def loader(train_split, validation_split, testing_split, ss=256):
    train_dataset = MovieDataset(train_split)
    validation_dataset = MovieDataset(validation_split)
    test_dataset = MovieDataset(testing_split)
    training_load = DataLoader(train_dataset, batch_size=ss, shuffle=True)
    validation_load = DataLoader(validation_dataset, batch_size=ss, shuffle=False)
    testing_load = DataLoader(test_dataset, batch_size=ss, shuffle=False)
    return training_load, validation_load, testing_load

def main():

    file_path = "../data/ml-1m/ratings.dat"
    ratings = load_data(str(file_path))
    print(ratings.head())
    final, total_users, total_movies = reindexing(ratings)
    point = get_positive_interactions(final)
    print(point.head())
    q1, q2, q3 = split_dataset(point)
    neint_tr = get_negativesampling(q1, point, total_movies, rat=4)
    neint_va = get_negativesampling(q2, point, total_movies, rat=4)
    neint_te = get_negativesampling(q3, point, total_movies, rat=4)
    train_split = add(q1, neint_tr)
    validation_split = add(q2, neint_va)
    testing_split = add(q3, neint_te)
    training_load, validation_load, testing_load = loader(train_split, validation_split, testing_split)
    print("Data preprocessing completed")
    print(f"Total users: {total_users} ")
    print(f"Total movies: {total_movies}")
    print(f"Training size - {len(train_split)} ")
    print(f"Validation size - {len(validation_split)} ")
    print(f"Test size - {len(testing_split)}")
    print("Train label distribution:", train_split['label'].value_counts().to_dict())
    return training_load, validation_load, testing_load, total_users, total_movies

if __name__ == "__main__":
    training_load, validation_load, testing_load, total_users, total_movies = main()
