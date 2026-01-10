import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.relu(self.dropout(attn_out)))
        x = self.norm2(x + self.dropout(self.relu(self.fc(x))))
        return x.squeeze(1)
    
class AttentionNet(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim, hidden_dims,
                 n_attention_blocks, dropout, num_heads=4, global_mean=0.0):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        self.feature_dim = hidden_dims[-1]
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

        self.global_mean = global_mean

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(embedding_dim * 2, num_heads, dropout)
            for _ in range(n_attention_blocks)
        ])

        layers = []
        dim = embedding_dim * 2
        for h in hidden_dims:
            layers += [
                nn.Linear(dim, h),
                nn.ReLU(),
                nn.LayerNorm(h),
                nn.Dropout(dropout)
            ]
            dim = h

        self.mlp_features = nn.Sequential(*layers)
        self.final_layer = nn.Linear(dim, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_ids, movie_ids, return_features=False):
        u = self.user_embedding(user_ids)
        m = self.movie_embedding(movie_ids)

        x = torch.cat([u, m], dim=-1)
        for block in self.attention_blocks:
            x = block(x)

        features = self.mlp_features(x)

        if return_features:
            return features

        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.movie_bias(movie_ids).squeeze(-1)

        out = self.final_layer(features).squeeze(-1)
        return out + bu + bi + self.global_mean