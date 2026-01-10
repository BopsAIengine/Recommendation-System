import torch
import torch.nn.functional as F

def info_nce_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    pos = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
    ttl = torch.exp(torch.matmul(z1, z2.t()) / temperature).sum(dim=1)
    return -torch.log(pos / ttl).mean()

def build_edge_index(df):
    users = torch.LongTensor(df['user_idx'].values)
    items = torch.LongTensor(df['movie_idx'].values)
    return torch.stack([users, items], dim=0)