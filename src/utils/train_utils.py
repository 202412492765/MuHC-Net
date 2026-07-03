import numpy as np
import torch


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CausalEffectEstimator:
    def __init__(self, n_modalities=4, device='cuda'):
        self.n_modalities = n_modalities
        self.device = device
        self.train_means = [None] * n_modalities

    def set_train_means(self, means_list):
        for i, m in enumerate(means_list):
            self.train_means[i] = torch.as_tensor(m, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def compute_ranking_effects(self, nets, fuse, inputs, H_list, t, e):
        outputs = [nets[i](inputs[i], H_list[i]) for i in range(self.n_modalities)]
        emb4 = [o[2] for o in outputs] if len(outputs[0]) == 3 else [o[1] for o in outputs]
        haz_f, _ = fuse(emb4)
        haz = haz_f.squeeze()

        effects = []
        for i in range(self.n_modalities):
            inputs_cf = [inp.clone() for inp in inputs]
            mean_i = self.train_means[i].view(1, -1).expand_as(inputs_cf[i])
            inputs_cf[i] = mean_i

            outputs_cf = [nets[j](inputs_cf[j], H_list[j]) for j in range(self.n_modalities)]
            emb4_cf = [o[2] for o in outputs_cf] if len(outputs_cf[0]) == 3 else [o[1] for o in outputs_cf]
            haz_f_cf, _ = fuse(emb4_cf)
            haz_cf = haz_f_cf.squeeze()

            t_i = t.unsqueeze(1)
            t_j = t.unsqueeze(0)
            e_i = e.unsqueeze(1)
            comparable = (t_i < t_j) & (e_i == 1)

            sign_orig = torch.sign(haz.unsqueeze(1) - haz.unsqueeze(0))
            sign_cf = torch.sign(haz_cf.unsqueeze(1) - haz_cf.unsqueeze(0))

            flips = (sign_orig != sign_cf) & comparable
            effect = flips.sum().float() / (comparable.sum().float() + 1e-7)
            effects.append(effect)

        return torch.tensor(effects, device=self.device)