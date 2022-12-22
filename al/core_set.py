import torch
import numpy as np
from scipy.spatial import distance_matrix
from .sampler import Sampler


class CoreSet(Sampler):
    """
    Implementation of CoreSet :footcite:`sener2018active` Strategy. A diversity-based
    approach using coreset selection. The embedding of each example is computed by the network’s
    penultimate layer and the samples at each round are selected using a greedy furthest-first
    traversal conditioned on all labeled examples.
    """

    name = "core_set"

    def furthest_first(self, unlabeled_embeddings, labeled_embeddings, n):
        m = unlabeled_embeddings.shape[0]
        if labeled_embeddings.shape[0] == 0:
            min_dist = torch.tile(float("inf"), m)
        else:
            dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
            min_dist = torch.min(dist_ctr, dim=1)[0]

        idxs = []

        for i in range(n):
            idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(
                unlabeled_embeddings, unlabeled_embeddings[[idx], :]
            )
            min_dist = torch.minimum(min_dist, dist_new_ctr[:, 0])

        return idxs

    def query(self, query_size, unlab_inds, lab_inds, model, **kwargs):
        embedding_unlabeled = self._forward_iter(unlab_inds, model.get_encoded)
        embedding_labeled = self._forward_iter(lab_inds, model.get_encoded)
        top_n = self.furthest_first(embedding_unlabeled, embedding_labeled, query_size)

        return unlab_inds[top_n]
