

import torch
import torch.nn as nn

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum() / weight.sum()

def decorrelation_loss(zA, zB):
    # Subtract mean per feature dimension
    zA = zA - zA.mean(dim=0, keepdim=True)
    zB = zB - zB.mean(dim=0, keepdim=True)
    
    # Covariance matrix (shared_dim × shared_dim)
    cov = (zA.T @ zB) / (zA.size(0) - 1)
    
    # Frobenius norm squared = sum of squared correlations
    return (cov ** 2).sum()

def full_decorr_loss(features):
    loss = 0.0
    num = len(features)
    for i in range(num):
        for j in range(i+1, num):
            loss += decorrelation_loss(features[i], features[j])
    return loss

class UncertaintyMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # s_t = log σ_t^2
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, preds, targets, weights):
        # preds:   (batch, num_tasks)
        # targets: (batch, num_tasks)

        loss = 0.0
        num_tasks = preds.size(1)

        for t in range(num_tasks):
            # precision = 1 / σ_t^2
            precision = torch.exp(-self.log_vars[t])

            # Mean squared error for task t
            mse_t = (preds[:, t] - targets[:, t])**2
            mse_t = mse_t.mean()

            # Kendall et al. regression loss
            task_loss = 0.5 * precision * mse_t + 0.5 * self.log_vars[t]

            loss += task_loss

        return loss