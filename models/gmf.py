import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim, global_mean=0.0):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

        self.affine_output = nn.Linear(embedding_dim, 1, bias=False)
        self.global_mean = global_mean

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        nn.init.xavier_uniform_(self.affine_output.weight)

    def forward(self, user_ids, movie_ids):
        u = self.user_embedding(user_ids)
        m = self.movie_embedding(movie_ids)

        interaction = self.affine_output(u * m).squeeze(-1)
        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.movie_bias(movie_ids).squeeze(-1)

        return interaction + bu + bi + self.global_mean