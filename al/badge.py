from .sampler import Sampler
import numpy as np

import torch
from torch import nn
from scipy import stats

from util import softmax


def init_centers(X, K, device):
    pdist = nn.PairwiseDistance(p=2)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    inds_all = [ind]
    cent_inds = [0.0] * len(X)
    cent = 0

    while len(mu) < K:
        if len(mu) == 1:
            D2 = pdist(
                torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device)
            )
            D2 = torch.flatten(D2)
            D2 = D2.cpu().numpy().astype(float)
        else:
            newD = pdist(
                torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device)
            )
            newD = torch.flatten(newD)
            newD = newD.cpu().numpy().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    cent_inds[i] = cent
                    D2[i] = newD[i]

        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        custom_dist = stats.rv_discrete(
            name="custom", values=(np.arange(len(D2)), Ddist)
        )
        ind = custom_dist.rvs(size=1)[0]
        mu.append(X[ind])
        inds_all.append(ind)
        cent += 1

    return inds_all


def reversed_centers(X, K, device):
    pdist = nn.PairwiseDistance(p=2)
    ind = np.argmin([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    inds_all = [ind]
    cent_inds = [0.0] * len(X)
    cent = 0

    while len(mu) < K:
        if len(mu) == 1:
            D2 = pdist(
                torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device)
            )
            D2 = torch.flatten(D2)
            D2 = D2.cpu().numpy().astype(float)
        else:
            newD = pdist(
                torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device)
            )
            newD = torch.flatten(newD)
            newD = newD.cpu().numpy().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    cent_inds[i] = cent
                    D2[i] = newD[i]

        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        custom_dist = stats.rv_discrete(
            name="custom", values=(np.arange(len(D2)), softmax(-Ddist))
        )
        ind = custom_dist.rvs(size=1)[0]
        mu.append(X[ind])
        inds_all.append(ind)
        cent += 1

    return inds_all


class BADGE(Sampler):
    """
    This method is based on the paper Deep Batch Active Learning by Diverse, Uncertain Gradient
    Lower Bounds `DBLP-Badge`. According to the paper, this strategy, Batch Active
    learning by Diverse Gradient Embeddings (BADGE), samples groups of points that are disparate
    and high magnitude when represented in a hallucinated gradient space, a strategy designed to
    incorporate both predictive uncertainty and sample diversity into every selected batch.
    Crucially, BADGE trades off between uncertainty and diversity without requiring any hand-tuned
    hyperparameters. Here at each round of selection, loss gradients are computed using the
    hypothesized labels. Then to select the points to be labeled are selected by applying
    k-means++ on these loss gradients.
    """

    name = "badge"

    def query(self, query_size, unlab_inds, model, criterion, num_targets, **kwargs):
        grad_embedding = self._get_grad_embedding(
            model, criterion, unlab_inds, num_targets, grad_embedding_type="linear"
        )
        grad_embedding = grad_embedding.cpu().detach().numpy()
        top_n = init_centers(grad_embedding, query_size, self.device)
        return unlab_inds[top_n]


class AntiBADGE(Sampler):
    """
    This method is based on the paper Deep Batch Active Learning by Diverse, Uncertain Gradient
    Lower Bounds `DBLP-Badge`. According to the paper, this strategy, Batch Active
    learning by Diverse Gradient Embeddings (BADGE), samples groups of points that are disparate
    and high magnitude when represented in a hallucinated gradient space, a strategy designed to
    incorporate both predictive uncertainty and sample diversity into every selected batch.
    Crucially, BADGE trades off between uncertainty and diversity without requiring any hand-tuned
    hyperparameters. Here at each round of selection, loss gradients are computed using the
    hypothesized labels. Then to select the points to be labeled are selected by applying
    k-means++ on these loss gradients.
    """

    name = "anti_badge"

    def query(self, query_size, unlab_inds, model, criterion, num_targets, **kwargs):
        grad_embedding = self._get_grad_embedding(
            model, criterion, unlab_inds, num_targets, grad_embedding_type="linear"
        )
        grad_embedding = grad_embedding.cpu().detach().numpy()
        top_n = reversed_centers(grad_embedding, query_size, self.device)
        return unlab_inds[top_n]
