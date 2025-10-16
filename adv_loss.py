import torch

def compute_mmd(z, z_prior=None):
    """
    Compute MMD between latent samples `z` and prior `z_prior` (default: N(0, I)).
    """
    if z_prior is None:
        z_prior = torch.randn_like(z)

    def _kernel(x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled = torch.exp(-torch.sum((x - y) ** 2, dim=2) / dim)
        return tiled

    xx = _kernel(z, z)
    yy = _kernel(z_prior, z_prior)
    xy = _kernel(z, z_prior)
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


from utils.losses import compute_mmd
...
z_sample = vae.encoder(x)[0]
mmd_loss = compute_mmd(z_sample)


import torch.nn.functional as F

def simclr_loss(z1, z2, temperature=0.1):
    """
    Computes the NT-Xent (SimCLR) contrastive loss between two sets of projections z1 and z2.
    """
    batch_size = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2N, D)

    similarity_matrix = torch.matmul(z, z.T)  # (2N, 2N)
    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z.device)

    return F.cross_entropy(logits / temperature, labels)



# assume x1 and x2 are different augmentations of the same input batch
z1 = vae.encoder(x1)[0]
z2 = vae.encoder(x2)[0]
contrastive_loss = simclr_loss(z1, z2)


import torch
from torch.autograd import grad

class EWC:
    def __init__(self, model, dataset, device='cuda'):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self._precision_matrices = self._diag_fisher(dataset)

    def _diag_fisher(self, dataset):
        model = self.model
        precision_matrices = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                precision_matrices[n] = torch.zeros_like(p)

        model.eval()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        for x in dataloader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(self.device)
            model.zero_grad()
            output = model.elbo(x, len(dataset))[0].mean()
            grads = grad(output, [p for n, p in model.named_parameters() if p.requires_grad], retain_graph=False)
            for (n, p), g in zip(model.named_parameters(), grads):
                if p.requires_grad:
                    precision_matrices[n] += g.data ** 2

        for n in precision_matrices:
            precision_matrices[n] /= len(dataloader)
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss


from continual.ewc import EWC
...
ewc = EWC(model=vae, dataset=train_loader.dataset, device='cuda')


ewc_loss = ewc.penalty(vae)

