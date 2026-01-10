import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from pipeline import RatingDataset

def eval_model(model, test_df, batch_size, device, min_rating, max_rating):
    test_dataset = RatingDataset(test_df['user_idx'].values, test_df['movie_idx'].values, test_df['rating'].values)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for users, movies, ratings in test_loader:
            users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
            predictions = model(users, movies)
            test_loss += criterion(predictions, ratings).item()
    test_loss /= len(test_loader)

    scaling_factor = max_rating - min_rating
    test_rmse_scaled = np.sqrt(test_loss)
    test_rmse = test_rmse_scaled * scaling_factor

    print(f"Test RMSE: {test_rmse:.4f}")
    return test_rmse

    
def sample(model, test_df, n_samples, device):
    if n_samples > len(test_df):
        n_samples = len(test_df)

    sample_df = test_df.sample(n=n_samples).copy()

    model.eval()
    with torch.no_grad():
        users = torch.LongTensor(sample_df['user_idx'].values).to(device)
        movies = torch.LongTensor(sample_df['movie_idx'].values).to(device)
        predictions = 1 + model(users, movies) * 4
        predictions = torch.clamp(predictions, 1.0, 5.0).cpu().numpy()

    sample_df['predicted_rating'] = predictions
    sample_df['rating'] = 1 + 4 * sample_df['rating']
    sample_df['error'] = sample_df['predicted_rating'] - sample_df['rating']
    sample_df['absolute_error'] = np.abs(sample_df['error'])

    sample_rmse = np.sqrt(np.mean(sample_df['error'] ** 2))
    sample_mae = np.mean(sample_df['absolute_error'])

    display_df = sample_df[['userId', 'movieId', 'rating', 'predicted_rating', 'error', 'absolute_error']].head(10)
    print(display_df.round(2).to_string(index=False))

    print("\n" + "-" * 100)
    print(f"RMSE: {sample_rmse:.4f}")
    print(f"MAE: {sample_mae:.4f}")

    return sample_df, sample_rmse