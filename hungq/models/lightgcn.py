import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_layers, edge_index, global_mean=0.0, learn_global_mean=False):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        if learn_global_mean:
            self.global_mean = nn.Parameter(torch.tensor(global_mean))
        else:
            self.register_buffer("global_mean", torch.tensor(global_mean))

        self.register_buffer("norm_adj", self.build_norm_adj(edge_index))

    def build_norm_adj(self, edge_index):
        device = edge_index.device
        num_nodes = self.n_users + self.n_items

        u = edge_index[0]
        i = edge_index[1] + self.n_users

        row = torch.cat([u, i])
        col = torch.cat([i, u])

        values = torch.ones(row.size(0), device=device)

        adj = torch.sparse_coo_tensor(
            torch.stack([row, col]),
            values,
            (num_nodes, num_nodes)
        ).coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        r, c = adj.indices()
        norm_values = deg_inv_sqrt[r] * adj.values() * deg_inv_sqrt[c]

        norm_adj = torch.sparse_coo_tensor(
            adj.indices(),
            norm_values,
            adj.size()
        )

        return norm_adj

    def propagate(self):
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )

        embeddings = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings.append(all_embeddings)

        final_embeddings = torch.mean(torch.stack(embeddings, dim=0), dim=0)
        users, items = torch.split(
            final_embeddings, [self.n_users, self.n_items]
        )

        return users, items

    def forward(self, user_ids, item_ids):
        user_emb, item_emb = self.propagate()

        u = user_emb[user_ids]
        i = item_emb[item_ids]

        interaction = (u * i).sum(dim=-1)

        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.item_bias(item_ids).squeeze(-1)

        return interaction + bu + bi + self.global_mean