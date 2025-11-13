import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta

        return grad_input, None

def grad_reverse(x, beta=1.0):
    return GradientReversalLayer.apply(x, beta)


class Adversary(nn.Module):
    def __init__(self, input_dim=2048*3, hidden_dim=128, num_sensitive_classes=2):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sensitive_classes)
        )
        self.grl = grad_reverse
    def forward(self, feature_representation, beta=1.0):
        reversed_feature = self.grl(feature_representation, beta)
        prediction = self.network(reversed_feature)
        return prediction


def adversarial_loss(feature_representation, sensitive_attributes, adversary_model, criterion_adv, beta=1.0):
    sensitive_attributes = sensitive_attributes.long().cuda()
    sensitive_pred = adversary_model(feature_representation, beta)
    loss_adv = criterion_adv(sensitive_pred, sensitive_attributes)

    return loss_adv