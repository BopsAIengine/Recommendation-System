import torch
import torch.nn as nn

class Ensemble(nn.Module):
    def __init__(self, models, learn_weights=True):
        super().__init__()

        self.models = nn.ModuleList(models)

        for m in self.models:
            for p in m.parameters():
                p.requires_grad = False

        n_models = len(models)

        if learn_weights:
            self.weights = nn.Parameter(torch.ones(n_models) / n_models)
        else:
            self.register_buffer("weights", torch.ones(n_models) / n_models)

    def forward(self, users, items):
        preds = []
        for model in self.models:
            preds.append(model(users, items))

        preds = torch.stack(preds, dim=0)
        weights = torch.softmax(self.weights, dim=0)

        return (weights[:, None] * preds).sum(dim=0)