import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from classifier_2 import LatentClassifier, train_classifier
import os
import torch


def compute_causal_contributions(z, y, device='cuda'):
    """
    Computes contribution of each dimension in z to predicting y.
    Uses logistic regression weights as a proxy.
    """
    z = z.to(device)
    y = y.to(device)
    input_dim = z.shape[1]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    z_np = scaler.fit_transform(z.cpu().numpy())
    y_np = y.cpu().numpy()

    clf = LogisticRegression(max_iter=1000).fit(z_np, y_np)
    contribution = np.abs(clf.coef_[0])  # shape: [latent_dim]
    return contribution  # shape: [latent_dim]


# def cluster_latents(contribution, num_clusters=3):
#     from sklearn.cluster import KMeans

#     # Contribution is shape [latent_dim] → reshape to [latent_dim, 1]
#     contrib_reshaped = contribution.reshape(-1, 1)

#     kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(contrib_reshaped)
#     cluster_ids = kmeans.labels_  # array of size [latent_dim] with cluster indices
#     return cluster_ids  # e.g., [0, 1, 1, 2, 0, ...]

# def cluster_latents(contribution, num_clusters=10, overlap_prob=0.3):
#     """
#     Cluster latent dimensions into overlapping clusters using KMeans + random overlap.
    
#     Args:
#         contribution (np.array): shape [latent_dim]
#         num_clusters (int): number of clusters
#         overlap_prob (float): probability that a dimension is added to an extra cluster

#     Returns:
#         clusters: list of lists; clusters[i] = list of latent dim indices in cluster i
#     """
#     from sklearn.cluster import KMeans

#     latent_dim = contribution.shape[0]
#     contrib_reshaped = contribution.reshape(-1, 1)
    
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(contrib_reshaped)
#     base_cluster_ids = kmeans.labels_  # shape [latent_dim]

#     # Initialize clusters from KMeans (non-overlapping)
#     clusters = [[] for _ in range(num_clusters)]
#     for dim, cid in enumerate(base_cluster_ids):
#         clusters[cid].append(dim)

#     # Now add overlap: with `overlap_prob`, assign each dim to extra cluster(s)
#     for dim in range(latent_dim):
#         for cid in range(num_clusters):
#             if cid != base_cluster_ids[dim] and np.random.rand() < overlap_prob:
#                 clusters[cid].append(dim)

#     # Optional: sort and deduplicate each cluster
#     clusters = [sorted(list(set(c))) for c in clusters]

#     return clusters  # List of lists of indices (overlapping allowed)

def cluster_latents(contribution, num_clusters=5, overlap_prob=0.3):
    """
    Cluster latent dimensions into overlapping clusters using KMeans + random overlap.
    If num_clusters > latent_dim, it is capped at latent_dim.
    """
    from sklearn.cluster import KMeans

    latent_dim = contribution.shape[0]
    contrib_reshaped = contribution.reshape(-1, 1)

    # Cap clusters at latent_dim
    num_clusters = min(num_clusters, latent_dim)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(contrib_reshaped)
    base_cluster_ids = kmeans.labels_

    clusters = [[] for _ in range(num_clusters)]
    for dim, cid in enumerate(base_cluster_ids):
        clusters[cid].append(dim)

    for dim in range(latent_dim):
        for cid in range(num_clusters):
            if cid != base_cluster_ids[dim] and np.random.rand() < overlap_prob:
                clusters[cid].append(dim)

    clusters = [sorted(list(set(c))) for c in clusters]

    return clusters


def train_cluster_classifiers(z, y, cluster_ids, num_clusters, save_dir, env_id, device='cuda'):
    classifiers = {}
    accuracies = {}

    for cluster_id in range(num_clusters):
        dim_mask = (cluster_ids == cluster_id)
        z_cluster = z[:, dim_mask]
        dataset = TensorDataset(z_cluster, y)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        classifier = LatentClassifier(input_dim=z_cluster.size(1)).to(device)
        classifier = train_classifier(loader, input_dim=z_cluster.size(1), device=device, epochs=2000)

        # Save classifier
        classifier_path = os.path.join(save_dir, f'classifier_cluster{cluster_id}_env{env_id}.pth')
        torch.save(classifier.state_dict(), classifier_path)
        classifiers[cluster_id] = classifier

        # Evaluate
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for z_batch, labels in loader:
                z_batch = z_batch.to(device)
                labels = labels.to(device)
                preds = classifier(z_batch).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Accuracy (env {env_id}) - Cluster {cluster_id}: {acc:.4f}")
        accuracies[cluster_id] = acc

    return classifiers, accuracies


def load_cluster_classifiers(env_id, cluster_ids, num_clusters, save_dir, input_dim, device='cuda'):
    classifiers = {}

    for cluster_id in range(num_clusters):
        classifier = LatentClassifier(input_dim=input_dim).to(device)
        path = os.path.join(save_dir, f'classifier_cluster{cluster_id}_env{env_id - 1}.pth')
        if os.path.exists(path):
            classifier.load_state_dict(torch.load(path))
            classifiers[cluster_id] = classifier
            print(f"Loaded classifier for cluster {cluster_id} from env {env_id - 1}")
        else:
            print(f"WARNING: Classifier path {path} not found")

    return classifiers

def train_classifier(dataloader, input_dim, device, epochs=2000, model=None):
    if model is None:
        model = LatentClassifier(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    return model


import torch
import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F

class ClusterClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout_prob=0.3):
        super(ClusterClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# def build_overlapping_clusters(causal_contribs, num_clusters=10, top_k_dims=5):
#     """
#     Build overlapping clusters from causal contribution scores.
#     Each cluster shares some latent dims.
#     """
#     latent_dim = causal_contribs.shape[0]  # assuming causal_contribs is a 1D tensor
#     clusters = {}

#     torch.manual_seed(42)  # for reproducibility

#     for cluster_id in range(num_clusters):
#         noise = torch.rand(latent_dim) * 0.05  # small noise to diversify cluster dims
#         noisy_scores = np.asarray(causal_contribs) + np.asarray(noise)

#         # noisy_scores = causal_contribs + noise

#         top_dims = torch.topk(noisy_scores, top_k_dims).indices.tolist()
#         clusters[cluster_id] = top_dims

#     return clusters

# def build_overlapping_clusters(causal_contribs, num_clusters=5, top_k_dims=5):
#     # Convert to tensor if needed
#     if isinstance(causal_contribs, np.ndarray):
#         causal_contribs = torch.tensor(causal_contribs, dtype=torch.float32)

#     # Add noise for variability
#     noise = torch.randn_like(causal_contribs) * 0.01
#     noisy_scores = causal_contribs + noise
    
#     print(f"noisy_scores shape: {noisy_scores.shape}, cluster_id: {cluster_id}")
#     print(f"noisy_scores[{cluster_id}] has shape: {noisy_scores[cluster_id].shape}")


#     clusters = {}
#     for cluster_id in range(num_clusters):
#         top_dims = torch.topk(noisy_scores[cluster_id], top_k_dims).indices.tolist()
#         clusters[cluster_id] = top_dims
#     return clusters

# def build_overlapping_clusters(causal_contribs, num_clusters=5, top_k_dims=5):
#     import torch
#     import numpy as np

#     # Ensure input is a Tensor
#     if isinstance(causal_contribs, np.ndarray):
#         causal_contribs = torch.tensor(causal_contribs, dtype=torch.float32)

#     latent_dim = causal_contribs.shape[0]

#     # Add small noise to break ties
#     noise = torch.randn_like(causal_contribs) * 0.01
#     noisy_scores = causal_contribs + noise

#     # Get top-k most contributing dims globally
#     top_global_dims = torch.topk(noisy_scores, min(num_clusters * top_k_dims, latent_dim)).indices.tolist()

#     # Create overlapping clusters
#     clusters = {}
#     for cluster_id in range(num_clusters):
#         # Randomly sample from top_global_dims to create overlapping clusters
#         cluster_dims = np.random.choice(top_global_dims, size=top_k_dims, replace=False)
#         clusters[cluster_id] = sorted(set(cluster_dims))  # remove duplicates if any

#     return clusters

from sklearn.cluster import KMeans
import numpy as np
import torch

def build_overlapping_clusters(causal_contribs, num_clusters=24, top_k_dims=8, overlap=8):
    print("num_clusters ", num_clusters, "  top_k_dims ", top_k_dims, "  overlap ", overlap)
    """
    Clusters latent dimensions using KMeans and adds overlap by including neighboring dimensions in sorted contribution.
    
    Args:
        causal_contribs: (latent_dim,) tensor or ndarray
        num_clusters: number of clusters to form
        top_k_dims: number of top dimensions per cluster (base size)
        overlap: number of additional nearby dims to add per cluster for overlap
    
    Returns:
        clusters: dict {cluster_id: list of latent dim indices}
    """

    # Ensure tensor
    if isinstance(causal_contribs, np.ndarray):
        causal_contribs = torch.tensor(causal_contribs, dtype=torch.float32)

    latent_dim = causal_contribs.shape[0]

    # Step 1: Prepare data for clustering
    contrib_reshaped = causal_contribs.view(-1, 1).cpu().numpy()

    # Cap number of clusters if needed
    num_clusters = min(num_clusters, latent_dim)

    # Step 2: KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(contrib_reshaped)

    # Step 3: Base clusters from KMeans
    clusters = {i: [] for i in range(num_clusters)}
    for dim_index, label in enumerate(labels):
        clusters[label].append(dim_index)

    # Step 4: Add soft overlap based on neighbor importance
    sorted_dims = torch.argsort(causal_contribs, descending=True).tolist()

    for cluster_id in clusters:
        base_dims = clusters[cluster_id]

        # For overlap: sample dims adjacent to base dims in importance ranking
        extra_dims = []
        for dim in base_dims:
            if dim in sorted_dims:
                idx = sorted_dims.index(dim)
                neighbors = sorted_dims[max(0, idx - overlap): idx] + \
                            sorted_dims[idx + 1: idx + 1 + overlap]
                extra_dims.extend(neighbors)

        # Merge base and extra, remove duplicates
        clusters[cluster_id] = sorted(set(base_dims + extra_dims))

    return clusters
