import torch
import torch.nn as nn
from .lightgcn import LightGCN

class LightGCNPP(LightGCN):
    def __init__(self, *args, residual=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual = residual
        self.layer_weights = nn.Parameter(
            torch.ones(self.n_layers + 1)
        )

    def propagate(self):
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )

        embeddings = [all_embeddings]

        for _ in range(self.n_layers):
            neigh = torch.sparse.mm(self.norm_adj, all_embeddings)
            all_embeddings = neigh + all_embeddings if self.residual else neigh
            embeddings.append(all_embeddings)

        stack = torch.stack(embeddings, dim=0)
        alpha = torch.softmax(self.layer_weights, dim=0)
        final_embeddings = torch.sum(alpha[:, None, None] * stack, dim=0)

        return torch.split(final_embeddings, [self.n_users, self.n_items])