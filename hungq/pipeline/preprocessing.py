from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import torch

class RatingDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = torch.LongTensor(users)
        self.movies = torch.LongTensor(movies)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]
    
def prepare_loaders(train_df, val_df, batch_size):
    train_dataset = RatingDataset(
        train_df['user_idx'].values,
        train_df['movie_idx'].values,
        train_df['rating'].values
    )
    val_dataset = RatingDataset(
        val_df['user_idx'].values,
        val_df['movie_idx'].values,
        val_df['rating'].values
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def preprocessing(data_path, test_size=0.2, batch_size=512, random_state=42):
    if not os.path.exists(data_path):
        print("Data file not found. Generating dummy data...")
        users = np.random.randint(0, 100, 10000)
        movies = np.random.randint(0, 200, 10000)
        ratings = np.random.randint(1, 6, 10000)
        df = pd.DataFrame({'userId': users, 'movieId': movies, 'rating': ratings})
    else:
        df = pd.read_csv(data_path, sep='\t', names=['userId', 'movieId', 'rating'])

    user_ids = df['userId'].unique()
    movie_ids = df['movieId'].unique()

    min_rating = df['rating'].min()
    max_rating = df['rating'].max()
    print(f"Scaling ratings from [{min_rating}, {max_rating}] to [0, 1].")
    df['rating'] = (df['rating'] - min_rating) / (max_rating - min_rating)

    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_map = {mid: idx for idx, mid in enumerate(movie_ids)}

    df['user_idx'] = df['userId'].map(user_map)
    df['movie_idx'] = df['movieId'].map(movie_map)

    n_users = len(user_map)
    n_movies = len(movie_map)

    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    val_size_relative = test_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size_relative, random_state=random_state)
    train_loader, val_loader = prepare_loaders(train_df, val_df, batch_size=batch_size)

    return train_df, val_df, test_df, train_loader, val_loader, user_map, movie_map, n_users, n_movies, min_rating, max_rating
