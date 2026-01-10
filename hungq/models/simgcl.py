import torch
import torch.nn.functional as F
from hungq.util import info_nce_loss
from .lightgcn import LightGCN

class SimGCL(LightGCN):
    def __init__(self, *args, eps=0.1, temperature=0.2, lambda_cl=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.temperature = temperature
        self.lambda_cl = lambda_cl

    def propagate_perturbed(self, eps=0.1):
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )

        embeddings = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            noise = F.normalize(torch.rand_like(all_embeddings), dim=-1)
            all_embeddings = all_embeddings + eps * noise
            embeddings.append(all_embeddings)

        final_embeddings = torch.mean(torch.stack(embeddings, dim=0), dim=0)
        return torch.split(final_embeddings, [self.n_users, self.n_items])

    def contrastive_loss(self, users, items):
        u1, i1 = self.propagate_perturbed(self.eps)
        u2, i2 = self.propagate_perturbed(self.eps)

        loss_u = info_nce_loss(u1[users], u2[users], self.temperature)
        loss_i = info_nce_loss(i1[items], i2[items], self.temperature)
        return loss_u + loss_i