import torch
import torch.nn as nn
import torch.nn.functional as F


def ranking_distill_loss_v2(haz_f, haz4, time, event):
    haz_f = haz_f.view(-1)
    N = len(time)
    if N < 2:
        return torch.tensor(0.0, device=haz_f.device)

    t_i = time.unsqueeze(1)
    t_j = time.unsqueeze(0)
    e_i = event.unsqueeze(1)
    comparable = (t_i < t_j) & (e_i == 1)

    if comparable.sum() < 1:
        return torch.tensor(0.0, device=haz_f.device)

    p_teacher = torch.sigmoid(haz_f.unsqueeze(1) - haz_f.unsqueeze(0))
    p_teacher = p_teacher[comparable].detach()

    loss = 0
    for c in range(4):
        p_student = torch.sigmoid(haz4[:, c].unsqueeze(1) - haz4[:, c].unsqueeze(0))
        p_student = p_student[comparable]
        loss += F.binary_cross_entropy(p_student, p_teacher, reduction='mean')
    return loss / 4


class ConcordLossV2(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, loghaz, time, event, haz4=None):
        loghaz = loghaz.view(-1)
        time, event = time.view(-1), event.view(-1)
        idx = torch.argsort(time, descending=True)
        loghaz, event = loghaz[idx], event[idx]
        gamma = torch.exp(loghaz)
        logcum = torch.log(torch.cumsum(gamma, dim=0) + 1e-7)
        loss_cox = -torch.sum((loghaz - logcum) * event)
        loss_cox = loss_cox / (event.sum() + 1e-7)

        if haz4 is None:
            return loss_cox

        loss_cons = 0
        for c in range(4):
            haz_c = haz4[:, c]
            gamma_c = torch.exp(haz_c[idx])
            logcum_c = torch.log(torch.cumsum(gamma_c, dim=0) + 1e-7)
            loss_cons -= torch.sum((haz_c[idx] - logcum_c) * event)
        loss_cons = loss_cons / (4 * (event.sum() + 1e-7))
        return loss_cox + self.alpha * loss_cons


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, haz4, time, event):
        time, event = time.view(-1), event.view(-1)
        idx = torch.argsort(time, descending=True)
        event = event[idx]
        loss_cons = 0
        for c in range(4):
            haz_c = haz4[:, c]
            gamma_c = torch.exp(haz_c[idx])
            logcum_c = torch.log(torch.cumsum(gamma_c, dim=0) + 1e-7)
            loss_cons -= torch.sum((haz_c[idx] - logcum_c) * event)
        loss_cons = loss_cons / (4 * (event.sum() + 1e-7))
        return loss_cons