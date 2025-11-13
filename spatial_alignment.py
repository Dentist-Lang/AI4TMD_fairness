import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_alignment_loss(preds, labels, sensitive_attributes, lambda_fairness=0.1):
    fairness_loss = 0
    batch_size = preds.size(0)

    overall_dist = torch.mean(preds, dim=0)

    unique_groups = torch.unique(sensitive_attributes, dim=0)
    group_dists = []
    for group in unique_groups:
        group_indices = (sensitive_attributes == group).all(dim=1).nonzero().flatten()
        group_preds = preds[group_indices]
        group_dist = torch.mean(group_preds, dim=0)
        group_dists.append(group_dist)

    sinkhorn_distances = [F.pairwise_distance(overall_dist, group_dist) for group_dist in group_dists]
    fairness_loss = torch.mean(torch.stack(sinkhorn_distances))

    return lambda_fairness * fairness_loss
