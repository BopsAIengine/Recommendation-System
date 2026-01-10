import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, mlp_hidden_dims=(128, 64), dropout=0.1, global_mean=0.0):
        super().__init__()

        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)

        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_mean = global_mean

        layers = []
        input_dim = embedding_dim * 2
        for h in mlp_hidden_dims:
            layers += [
                nn.Linear(input_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            input_dim = h
        self.mlp = nn.Sequential(*layers)

        self.output_layer = nn.Linear(embedding_dim + input_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [
            self.user_embedding_gmf,
            self.item_embedding_gmf,
            self.user_embedding_mlp,
            self.item_embedding_mlp
        ]:
            nn.init.normal_(emb.weight, std=0.01)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, user_ids, item_ids):
        u_gmf = self.user_embedding_gmf(user_ids)
        i_gmf = self.item_embedding_gmf(item_ids)
        gmf_out = u_gmf * i_gmf

        u_mlp = self.user_embedding_mlp(user_ids)
        i_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_input)

        fusion = torch.cat([gmf_out, mlp_out], dim=-1)
        pred = self.output_layer(fusion).squeeze(-1)

        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.item_bias(item_ids).squeeze(-1)

        return pred + bu + bi + self.global_mean