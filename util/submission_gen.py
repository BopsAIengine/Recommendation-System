import pandas as pd
import numpy as np
import torch

def check_coverage(movie_map, user_map, prompt_path):
    prompt_df = pd.read_csv(prompt_path, sep='\t', names=['UserId', 'MovieId'])

    missing_users = set(prompt_df['UserId']) - set(user_map.keys())
    missing_movies = set(prompt_df['MovieId']) - set(movie_map.keys())

    ok = True
    if missing_users:
        print(f"Missing {len(missing_users)} users not found in user_map:")
        print(list(missing_users)[:10], "..." if len(missing_users) > 10 else "")
        ok = False

    if missing_movies:
        print(f"Missing {len(missing_movies)} movies not found in movie_map:")
        print(list(missing_movies)[:10], "..." if len(missing_movies) > 10 else "")
        ok = False

    if ok:
        print("All users and movies are covered.")

    return ok

def generate_submission(movie_map, user_map, model, prompt_path, output_path="submission.csv", device='cuda'):
    prompt_df = pd.read_csv(prompt_path, sep='\t', names=['UserId', 'MovieId'])
    prompt_df['UserIdx'] = prompt_df['UserId'].map(user_map)
    prompt_df['MovieIdx'] = prompt_df['MovieId'].map(movie_map)

    model.eval()
    model = model.to(device)

    preds = []
    with torch.no_grad():
        for _, row in prompt_df.iterrows():
            user_idx = row['UserIdx']
            movie_idx = row['MovieIdx']

            if pd.notna(movie_idx):
                user_tensor = torch.LongTensor([user_idx]).to(device)
                movie_tensor = torch.LongTensor([movie_idx]).to(device)
                output = model(user_tensor, movie_tensor).item()
                pred = np.clip(1 + output * 4, 1.0, 5.0)
            else:
                pred = 4.0

            preds.append(pred)

    submission = pd.DataFrame({
        "Id": np.arange(1, len(preds) + 1),
        "Score": preds
    })

    submission.to_csv(output_path, index=False)
    print(f"Saved submission file to: {output_path}")
    print(submission.head())

    return submission
