import os
import time
import math
from numbers import Number
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import lib.dist as dist
import lib.utils as utils
from lib.full_unconfound import UnconfoundedDataset
import lib.full_dataset_loading as dset
from lib.full_dataset_loading import collect_samples
from lib.flows import FactorialNormalizingFlow
from causal_attributions import compute_causal_contributions, build_overlapping_clusters,cluster_latents, train_cluster_classifiers, load_cluster_classifiers, train_classifier, ClusterClassifier
import numpy as np

import torch.nn.functional as F
from elbo_decomposition import elbo_decomposition
from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces, plot_vs_gt_concon  # noqa: F401
from classifier_2 import LatentClassifier, train_classifier

class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class VAE(nn.Module):
    def __init__(self, z_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, conv=False, mss=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0.6
        self.beta = 16
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)

        if use_cuda:
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    ##oriinal one
    def encode(self, x):
        x = x.view(x.size(0), 1, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params
    #original one


    def sample_positive_pairs(self, x):
        """
        Generates positive pairs of latent vectors by applying two stochastic encodings of x.
        Typically used for contrastive SimCLR-style training.
        """
        # Encode the same input twice to get different augmentations via stochastic sampling
        z_i, _ = self.encode(x)
        z_j, _ = self.encode(x)
        return z_i, z_j

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params
    
    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()
    
    def elbo(self, x, dataset_size, return_latents=False):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')
        l1_loss = F.l1_loss(x_recon, x, reduction='mean')
        
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            if return_latents:
                return elbo, elbo.detach(), zs
            else:
                return elbo, elbo.detach()

        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        if return_latents:
            return modified_elbo, elbo.detach(), zs
        else:
            return modified_elbo, elbo.detach()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


from torch.utils.data import DataLoader, Subset
# import dset  # make sure this is the correct import


from torch.utils.data import Subset


def setup_data_loaders(args, use_cuda=True):
    if args.dataset == 'concon':
        print("concon is being used.")
        
        base_path = "/path"
        env_dirs = {
            0: os.path.join(base_path, "t0"),
            1: os.path.join(base_path, "t1"),
            2: os.path.join(base_path, "t2"),
        }

        all_samples = []

        for env_id, dir_path in env_dirs.items():
            if not os.path.exists(dir_path):
                print(f"[WARNING] Skipping missing dir: {dir_path}")
                continue

            # ✅ Collect valid image samples
            samples = collect_samples(env_id, dir_path)

            # ✅ Add mixed samples
            if env_id == 1:
                samples += collect_samples(0, env_dirs[0], max_samples_per_class=50)
            elif env_id == 2:
                samples += collect_samples(0, env_dirs[0], max_samples_per_class=20)
                samples += collect_samples(1, env_dirs[1], max_samples_per_class=30)

            all_samples += samples

        print(f"Total training samples collected: {len(all_samples)}")
        
        # ✅ Build dataset with actual samples (img_path, label, env_id)
        dataset = dset.Concon(all_samples)

        train_loader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=use_cuda)
        return train_loader

    
def setup_data_loader_for_env(args, env_id, use_cuda=True):
    if args.dataset == 'concon':
        base_path = "/path/"
        env_dir = os.path.join(base_path, f"t{env_id}")
        samples = collect_samples(env_id, env_dir)

        # ✅ Add mixed samples for Env 1 and 2
        if env_id == 1:
            # Mix in 100 from env 0
            samples += collect_samples(0, os.path.join(base_path, "t0"), max_samples_per_class=100)
        elif env_id == 2:
            # Mix in 40 from env 0 and 60 from env 1
            samples += collect_samples(0, os.path.join(base_path, "t0"), max_samples_per_class=40)
            samples += collect_samples(1, os.path.join(base_path, "t1"), max_samples_per_class=60)

        # ✅ Shuffle and load
        dataset = dset.Concon(samples)
        train_loader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=use_cuda)

        print(f"[Env {env_id}] Final sample count (including mixed): {len(dataset)}")

        # Optional: print label distribution
        labels = [label for _, label, _ in dataset.samples]
        print(f"Label distribution for env {env_id}: 0 → {labels.count(0)}, 1 → {labels.count(1)}")

        return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat([
        test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)
    win_test_reco = vis.images(
        list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()), 10, 2,
        opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = list(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500
    else:
        warmup_iter = 10000

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


#### ORIGINAL ONE ###########
def get_latent_dataloader(vae, dataset_loader, batch_size=64):
    vae.eval()
    zs_list = []
    labels_list = []

    with torch.no_grad():
        for x, y, env_id in dataset_loader:

        # for x, y in dataset_loader:
            x = x.cuda()
            z, _ = vae.encode(x)
            zs_list.append(z.cpu())
            labels_list.append(y)

    zs_all = torch.cat(zs_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    dataset = TensorDataset(zs_all, labels_all)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


import torch
import torch.nn.functional as F

def compute_simclr_loss(z_i, z_j, temperature=0.8):
    batch_size = z_i.size(0)
    device = z_i.device

    # Normalize
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    # Cosine similarity matrix
    sim = torch.matmul(z, z.T) / temperature

    # Remove self-similarity
    mask = torch.eye(2 * batch_size, device=device).bool()
    sim = sim.masked_fill(mask, float('-inf'))

    # Create positive indices
    positive_indices = (torch.arange(2 * batch_size, device=device) + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim, positive_indices)
    return loss



import argparse
import time
import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


def gaussian_kernel_gpu_chunked(a, b, chunk_size=128):
    z_dim = a.size(1)
    result = []

    for i in range(0, a.size(0), chunk_size):
        a_chunk = a[i:i+chunk_size]
        diff = a_chunk.unsqueeze(1) - b.unsqueeze(0)  # [chunk, b_size, z_dim]
        dist_sq = torch.sum(diff ** 2, dim=-1)  # [chunk, b_size]
        kernel = torch.exp(-dist_sq / z_dim)
        result.append(kernel)

    return torch.cat(result, dim=0)

def compute_mmd_loss(z, z_prev, chunk_size=256):
    K_xx = gaussian_kernel_gpu_chunked(z, z, chunk_size)
    K_yy = gaussian_kernel_gpu_chunked(z_prev, z_prev, chunk_size)
    K_xy = gaussian_kernel_gpu_chunked(z, z_prev, chunk_size)

    m = z.size(0)
    n = z_prev.size(0)

    return (K_xx.sum() / (m * m) +
            K_yy.sum() / (n * n) -
            2 * K_xy.sum() / (m * n))


def get_latent_dataloader_uncon(vae, dataset_loader, batch_size=64):
    vae.eval()
    zs_list = []
    labels_list = []


    with torch.no_grad():
        for x, y in dataset_loader:


        # for x, y in dataset_loader:
            x = x.cuda()
            z, _ = vae.encode(x)
            zs_list.append(z.cpu())
            labels_list.append(y)


    zs_all = torch.cat(zs_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)


    dataset = TensorDataset(zs_all, labels_all)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def setup_data_loaders_uncon(args, use_cuda=True):
    """
    Returns the unconfounded data loader. This is used in evaluation after all environments are trained.
    """
    print("Loading unconfounded dataset...")
    dataset = UnconfoundedDataset()

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, drop_last=False, **kwargs)

    print(f"Total unconfounded samples: {len(dataset)}")
    return loader




def evaluate_on_unconfounded(vae, cluster_classifiers, clusters, args, device):
    print("\n--- Evaluating on Unconfounded Data ---")
    unconfounded_loader = setup_data_loaders_uncon(args, use_cuda=True)

    
    all_z_uc, all_y_uc = [], []
    vae.eval()
    with torch.no_grad():
        for batch in unconfounded_loader:
            x_uc = batch[0].to(device)
            y_uc = batch[1].to(device)
            z_uc = vae.encoder(x_uc)  # Encoder output should be latent vectors
            all_z_uc.append(z_uc.cpu())
            all_y_uc.append(y_uc.cpu())

    all_z_uc = torch.cat(all_z_uc, dim=0)
    all_y_uc = torch.cat(all_y_uc, dim=0)

    # Pass through each cluster classifier
    for cluster_id, clf in cluster_classifiers.items():
        clf.eval()
        cluster_dims = clusters[cluster_id]
        z_cluster_uc = all_z_uc[:, cluster_dims].to(device)
        y_uc = all_y_uc.to(device)
        with torch.no_grad():
            outputs = clf(z_cluster_uc)
            preds = outputs.argmax(dim=1)
            acc = (preds == y_uc).float().mean().item()
        print(f"Cluster {cluster_id} accuracy on unconfounded data: {acc:.4f}")

MMD_FREQ = 3       # Only compute MMD every 5 iterations
MAX_MMD_SAMPLES = 1000  


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.alpha = 1.0
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 3)  # 3 domains/environments
        )

    def forward(self, z):
        z_reversed = GradientReverse.apply(z)  # Gradient Reversal Layer
        return self.net(z_reversed)

def irm_penalty(logits, labels):
    scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss = F.cross_entropy(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


class EnvClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=2, dropout_prob=0.3):
        super(EnvClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        return self.classifier(x)


def main():
    overall_start_time = time.time()

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='shapes', type=str,
                        choices=['shapes', 'faces', 'concon'])
    parser.add_argument('--env_id', type=int, default=4, help='Environment ID for continual training')
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=3, type=int)
    parser.add_argument('-z', '--latent-dim', default=28, type=int)
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--beta', default=40.0, type=float)
    parser.add_argument('--lambda_mmd', default=100.0, type=float)
    parser.add_argument('--lambda_simclr', default=40.0, type=float)
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true')
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--save', default='alpha_test1')
    parser.add_argument('--log_freq', default=40, type=int)
    parser.add_argument('--tag', default='default', type=str)

    args = parser.parse_args()
    
    import os
    import json

    # Ensure directory exists
    os.makedirs(args.save, exist_ok=True)

    # Save args to file
    args_file = os.path.join(args.save, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Also print to console and log file
    print("Command-line arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Setup saving directory
    if not args.save.endswith('/'):
        args.save = f"{args.save}/"
    args.save = os.path.join(args.save, "alpha")
    os.makedirs(args.save, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)

    if args.dist == 'normal':
        prior_dist, q_dist = dist.Normal(), dist.Normal()
    elif args.dist == 'laplace':
        prior_dist, q_dist = dist.Laplace(), dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
              include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss)
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)


    domain_discriminator = DomainDiscriminator(input_dim=args.latent_dim).to(device)
    domain_optimizer = optim.Adam(domain_discriminator.parameters(), lr=1e-3)

    env_classifier = EnvClassifier(input_dim=args.latent_dim).to(device)
    clf_optimizer = optim.Adam(env_classifier.parameters(), lr=1e-3)



    if args.visdom:
        vis = visdom.Visdom(env=args.save, port=4500)

    latent_env_store = {}
    cluster_info_per_env = {}
    classifiers_per_env = {}
    cluster_classifiers = None
    cluster_optimizers = {}

    for env_id in [0, 1, 2]:
        print(f"\n--- Training on Environment {env_id} ---\n")
        env_start_time = time.time()
        args.env_id = env_id
        train_loader = setup_data_loader_for_env(args, env_id, use_cuda=True)

        # Load encoder/decoder from previous env
        if env_id > 0:
            prev_encoder_path = os.path.join(args.save, f'encoder_env{env_id - 1}.pth')
            if os.path.exists(prev_encoder_path):
                vae.encoder.load_state_dict(torch.load(prev_encoder_path, map_location=device))

            prev_decoder_path = os.path.join(args.save, f'decoder_env{env_id - 1}.pth')
            if os.path.exists(prev_decoder_path):
                vae.decoder.load_state_dict(torch.load(prev_decoder_path, map_location=device))

        dataset_size = len(train_loader.dataset)
        print("total sample in one env is ", dataset_size)
        iteration = 1
        pbar = tqdm(total=len(train_loader) * args.num_epochs, desc=f"Training Env {env_id}")
        for epoch in range(args.num_epochs):
            for i, batch in enumerate(train_loader):
                pbar.update(1)
                vae.train()
                anneal_kl(args, vae, iteration)
                iteration += 1

                optimizer.zero_grad()

                # Unpack and convert to tensors
                print(f"[DEBUG] Batch type: {type(batch)}")
                
                print(f"[DEBUG] Batch[0]: {batch[0]}")
                x_list, y_list, _ = zip(*batch)  # discard env_id if unused
                x = torch.stack(x_list).to(device)
                y = torch.tensor(y_list).to(device)

                obj, elbo, z = vae.elbo(x, dataset_size, return_latents=True)

                if utils.isnan(obj).any():
                    raise ValueError('NaN in objective.')
                latent_env_store.setdefault(env_id, []).append(z.detach().cpu())

                mmd_loss = torch.tensor(0.0, device=z.device)
                simclr_loss = torch.tensor(0.0, device=z.device)

                if env_id > 0 and iteration % MMD_FREQ == 0:
                    z_prev = torch.cat(latent_env_store[env_id - 1], dim=0).to(z.device)

                    if z_prev.size(0) > MAX_MMD_SAMPLES:
                        idx = torch.randperm(z_prev.size(0))[:MAX_MMD_SAMPLES]
                        z_prev = z_prev[idx]

                    mmd_loss = compute_mmd_loss(z, z_prev)

                    if hasattr(vae, 'sample_positive_pairs'):
                        z_i, z_j = vae.sample_positive_pairs(x)
                        simclr_loss = compute_simclr_loss(z_i, z_j)

                domain_labels = torch.full((z.size(0),), env_id, dtype=torch.long, device=z.device)
                domain_preds = domain_discriminator(z.detach())
                domain_loss = F.cross_entropy(domain_preds, domain_labels)

                domain_optimizer.zero_grad()
                domain_loss.backward()
                domain_optimizer.step()

                clf_logits = env_classifier(z)
                ce_loss = F.cross_entropy(clf_logits, y)
                irm_loss = irm_penalty(clf_logits, y)

                total_loss = obj.mean() + args.lambda_mmd * mmd_loss + args.lambda_simclr * simclr_loss
                total_loss += ce_loss + irm_loss
                total_loss.mul(-1).backward()
                optimizer.step()
        pbar.close()
     
        # Get latent embeddings
        dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False)
        all_z, all_y = [], []
        for z_batch, y_batch in get_latent_dataloader(vae, dataset_loader):
            all_z.append(z_batch.cpu())
            all_y.append(y_batch.cpu())
        all_z = torch.cat(all_z, 0)
        all_y = torch.cat(all_y, 0)

        causal_contribs = compute_causal_contributions(all_z, all_y, device)

   
        if env_id == 0:
            clusters = build_overlapping_clusters(causal_contribs, num_clusters=8, top_k_dims=18)

            if isinstance(clusters, np.ndarray):
                cluster_ids = clusters
                clusters = {}
                for i, cluster_id in enumerate(cluster_ids):
                    clusters.setdefault(cluster_id, []).append(i)

            cluster_info_per_env[0] = clusters
            cluster_classifiers = {
                i: ClusterClassifier(input_dim=len(cluster)).to(device)
                for i, cluster in clusters.items()

            
            }
            cluster_optimizers = {
                i: torch.optim.Adam(clf.parameters(), lr=1e-3)
                for i, clf in cluster_classifiers.items()
            }
        else:
            clusters = cluster_info_per_env[0]

        # Fine-tune classifiers
        for epoch in range(args.num_epochs):
            for i, (z_batch, y_batch) in enumerate(DataLoader(list(zip(all_z, all_y)), batch_size=args.batch_size, shuffle=True)):
                z_batch = z_batch.to(device)
                y_batch = y_batch.to(device)
                for cluster_id, cluster_dims in clusters.items():
                    z_cluster = z_batch[:, cluster_dims]  # cluster_dims is a list of indices
                    clf = cluster_classifiers[cluster_id]
                    optimizer = cluster_optimizers[cluster_id]
                    outputs = clf(z_cluster)
                    loss = F.cross_entropy(outputs, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

      

        for cluster_id, clf in cluster_classifiers.items():
            torch.save(clf.state_dict(), os.path.join(args.save, f'classifier_env{env_id}_cluster{cluster_id}.pth'))

        print(f"\nEvaluating classifiers on current env latents...")
        for cluster_id, clf in cluster_classifiers.items():
            clf.eval()
            cluster_dims = clusters[cluster_id]
            z_cluster = all_z[:, cluster_dims].to(device)
            y_labels = all_y.to(device)
            with torch.no_grad():
                outputs = clf(z_cluster)
                preds = outputs.argmax(dim=1)
                acc = (preds == y_labels).float().mean().item()
            print(f"Cluster {cluster_id} accuracy: {acc:.4f}")

    # ✅ Evaluate on unconfounded dataset
    print("\n--- Final Evaluation on Unconfounded Data ---")
    evaluate_on_unconfounded(vae, cluster_classifiers, clusters, args, device)
    
    end_time = time.time()
    elapsed_time =  end_time - overall_start_time

# Convert to hours, minutes, seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"Total execution time: {hours}h {minutes}m {seconds:.2f}s")
    

    return vae


if __name__ == '__main__':
    model = main()
    print("THE TRAINING HAS ENDED")
