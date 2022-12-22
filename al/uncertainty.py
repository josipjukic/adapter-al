from .sampler import Sampler


import numpy as np

from dataloaders import make_sklearn_dataset
from util import softmax


class LeastConfidentSampler(Sampler):
    name = "least_confident"

    def query(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()
        max_probs = np.max(probs, axis=1)

        top_n = np.argpartition(max_probs, query_size)[:query_size]
        return unlab_inds[top_n]


class MostConfidentSampler(Sampler):
    name = "most_confident"

    def query(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()
        max_probs = np.max(probs, axis=1)

        top_n = np.argpartition(max_probs, -query_size)[-query_size:]
        return unlab_inds[top_n]


class LeastConfidentDropoutSampler(Sampler):
    name = "least_confident_dropout"

    def __init__(self, dataset, batch_size, device, n_drop=10):
        self.n_drop = n_drop
        super().__init__(dataset, batch_size, device)

    def query(self, query_size, unlab_inds, model, num_labels, **kwargs):
        probs = (
            self._predict_probs_dropout(model, self.n_drop, unlab_inds, num_labels)
            .cpu()
            .numpy()
        )
        max_probs = np.max(probs, axis=1)

        top_n = np.argpartition(max_probs, query_size)[:query_size]
        return unlab_inds[top_n]


class MarginSampler(Sampler):
    name = "margin"

    def query(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()

        sort_probs = np.sort(probs, 1)[:, -2:]
        min_margin = sort_probs[:, 1] - sort_probs[:, 0]

        top_n = np.argpartition(min_margin, query_size)[:query_size]
        return unlab_inds[top_n]


class AntiMarginSampler(Sampler):
    name = "anti_margin"

    def query(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()

        sort_probs = np.sort(probs, 1)[:, -2:]
        min_margin = sort_probs[:, 1] - sort_probs[:, 0]

        top_n = np.argpartition(min_margin, -query_size)[-query_size:]
        return unlab_inds[top_n]


class MarginDropoutSampler(Sampler):
    name = "margin_dropout"

    def __init__(self, dataset, batch_size, device, n_drop=10):
        self.n_drop = n_drop
        super().__init__(dataset, batch_size, device)

    def query(self, query_size, unlab_inds, model, num_labels, **kwargs):
        probs = (
            self._predict_probs_dropout(model, self.n_drop, unlab_inds, num_labels)
            .cpu()
            .numpy()
        )
        sort_probs = np.sort(probs, 1)[:, -2:]
        min_margin = sort_probs[:, 1] - sort_probs[:, 0]

        top_n = np.argpartition(min_margin, query_size)[:query_size]
        return unlab_inds[top_n]


class EntropySampler(Sampler):
    name = "entropy"

    def query(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)

        top_n = np.argpartition(entropies, -query_size)[-query_size:]
        return unlab_inds[top_n]

    def weights(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)
        return entropies


class AntiEntropySampler(Sampler):
    name = "anti_entropy"

    def query(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)

        top_n = np.argpartition(entropies, query_size)[:query_size]
        return unlab_inds[top_n]

    def weights(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)
        return -entropies


class EntropyDropoutSampler(Sampler):
    name = "entropy_dropout"

    def __init__(self, dataset, batch_size, device, n_drop=30):
        self.n_drop = n_drop
        super().__init__(dataset, batch_size, device)

    def query(self, query_size, unlab_inds, model, num_labels, **kwargs):
        probs = (
            self._predict_probs_dropout(model, self.n_drop, unlab_inds, num_labels)
            .cpu()
            .numpy()
        )

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)

        top_n = np.argpartition(entropies, -query_size)[-query_size:]
        return unlab_inds[top_n]

    def weigths(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)
        return entropies


class MarginSklearn(Sampler):
    name = "margin_sklearn"

    def query(self, query_size, unlab_inds, model, vectorizer, **kwargs):
        X, _ = make_sklearn_dataset(self.dataset, vectorizer, indices=unlab_inds)
        probs = model.predict_proba(X)

        sort_probs = np.sort(probs, 1)[:, -2:]
        min_margin = sort_probs[:, 1] - sort_probs[:, 0]

        top_n = np.argpartition(min_margin, query_size)[:query_size]
        return unlab_inds[top_n]


class EntropySklearn(Sampler):
    name = "entropy_sklearn"

    def query(self, query_size, unlab_inds, model, vectorizer, **kwargs):
        X, _ = make_sklearn_dataset(self.dataset, vectorizer, indices=unlab_inds)
        probs = model.predict_proba(X)

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)

        top_n = np.argpartition(entropies, -query_size)[-query_size:]
        return unlab_inds[top_n]


class AntiEntropySklearn(Sampler):
    name = "anti_entropy_sklearn"

    def query(self, query_size, unlab_inds, model, vectorizer, **kwargs):
        X, _ = make_sklearn_dataset(self.dataset, vectorizer, indices=unlab_inds)
        probs = model.predict_proba(X)

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)

        top_n = np.argpartition(entropies, query_size)[:query_size]
        return unlab_inds[top_n]
