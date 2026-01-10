import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, gmf_model, attn_model, dropout=0.1, global_mean=0.0):
        super().__init__()

        self.gmf = gmf_model
        self.attn_net = attn_model
        self.global_mean = global_mean

        self.user_bias = gmf_model.user_bias
        self.movie_bias = gmf_model.movie_bias

        gmf_dim = gmf_model.user_embedding.embedding_dim
        attn_dim = attn_model.feature_dim

        self.attn_layer = nn.MultiheadAttention(
            embed_dim=gmf_dim + attn_dim,
            num_heads=4,
            batch_first=True
        )

        self.fusion_layer = nn.Linear(gmf_dim + attn_dim, 1)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fusion_layer.weight)

    def forward(self, user_ids, movie_ids):
        u = self.gmf.user_embedding(user_ids)
        m = self.gmf.movie_embedding(movie_ids)
        gmf_vec = u * m

        attn_vec = self.attn_net(user_ids, movie_ids, return_features=True)

        x = torch.cat([gmf_vec, attn_vec], dim=-1).unsqueeze(1)
        x, _ = self.attn_layer(x, x, x)
        x = self.dropout(x.squeeze(1))

        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.movie_bias(movie_ids).squeeze(-1)

        out = self.fusion_layer(x).squeeze(-1)
        return out + bu + bi + self.global_mean