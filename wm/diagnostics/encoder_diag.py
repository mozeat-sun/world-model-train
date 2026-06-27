"""Encoder diagnostics: PCA, probe AUC, KL validity, z consistency."""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


def compute_silhouette(z: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score of z clustering by market labels (e.g. volatility regime)."""
    if len(np.unique(labels)) < 2: return 0.0
    return silhouette_score(z, labels)


def probe_auc(z: np.ndarray, forward_ret: np.ndarray) -> float:
    """Linear probe: can a linear layer predict return direction from z?"""
    y = (forward_ret > 0).astype(int)
    if len(np.unique(y)) < 2: return 0.5
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z, y)
    pred = clf.predict_proba(z)[:, 1]
    return roc_auc_score(y, pred)


def z_consistency(z_train: np.ndarray, z_infer: np.ndarray) -> float:
    """Median cosine similarity between training z (VAE) and inference z (deterministic)."""
    cos = np.sum(z_train * z_infer, axis=-1) / (
        np.linalg.norm(z_train, axis=-1) * np.linalg.norm(z_infer, axis=-1) + 1e-8)
    return float(np.median(cos))


def kl_validity(z: np.ndarray) -> dict:
    """Check KL regularization: per-dimension std distribution."""
    stds = np.std(z, axis=0)
    return {
        'mean_std': float(np.mean(stds)),
        'min_std': float(np.min(stds)),
        'max_std': float(np.max(stds)),
        'pct_below_01': float(np.mean(stds < 0.1)),
        'pct_above_30': float(np.mean(stds > 3.0)),
    }


def temporal_smoothness(z_seq: np.ndarray) -> float:
    """Median cosine similarity between adjacent time steps."""
    cos_sims = []
    for t in range(len(z_seq) - 1):
        cos = np.dot(z_seq[t], z_seq[t+1]) / (
            np.linalg.norm(z_seq[t]) * np.linalg.norm(z_seq[t+1]) + 1e-8)
        cos_sims.append(cos)
    return float(np.median(cos_sims)) if cos_sims else 0.0


def run_encoder_diagnostics(encoder, dataloader, device='cpu') -> dict:
    """Run full encoder diagnostic suite."""
    encoder.eval()
    z_infer_list, z_train_list, ret_list = [], [], []

    with torch.no_grad():
        for obs, mask, fwd_ret in dataloader:
            obs, mask = obs.to(device), mask.to(device)
            z, mu, _ = encoder(obs, mask, deterministic=False)
            z_det, _, _ = encoder(obs, mask, deterministic=True)
            z_train_list.append(mu.cpu().numpy())
            z_infer_list.append(z_det.cpu().numpy())
            ret_list.append(fwd_ret.numpy())

    z_train = np.concatenate(z_train_list)
    z_infer = np.concatenate(z_infer_list)
    returns = np.concatenate(ret_list).mean(axis=-1)  # average across symbols

    return {
        'probe_auc': probe_auc(z_infer, returns),
        'z_consistency': z_consistency(z_train, z_infer),
        'kl_validity': kl_validity(z_train),
        'temporal_smoothness': temporal_smoothness(z_infer),
    }
