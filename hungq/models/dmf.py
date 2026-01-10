import torch
import torch.nn as nn

class DMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, user_hidden_dims, item_hidden_dims, dropout, global_mean=0.0):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_mean = global_mean

        u_layers = []
        in_dim = embedding_dim
        for h in user_hidden_dims:
            u_layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        self.user_mlp = nn.Sequential(*u_layers)

        i_layers = []
        in_dim = embedding_dim
        for h in item_hidden_dims:
            i_layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        self.item_mlp = nn.Sequential(*i_layers)
        self.user_proj = nn.Linear(user_hidden_dims[-1], embedding_dim)
        self.item_proj = nn.Linear(item_hidden_dims[-1], embedding_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)

        u = self.user_mlp(u)
        i = self.item_mlp(i)

        u = self.user_proj(u)
        i = self.item_proj(i)

        interaction = (u * i).sum(dim=-1)

        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.item_bias(item_ids).squeeze(-1)

        return interaction + bu + bi + self.global_mean