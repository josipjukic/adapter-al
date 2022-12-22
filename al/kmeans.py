from .sampler import Sampler

import numpy as np
from sklearn.cluster import KMeans


class KMeansSampler(Sampler):
    name = "kmeans"

    def query(self, query_size, unlab_inds, model, **kwargs):
        embeddings = self._forward_iter(unlab_inds, model.get_encoded).cpu().numpy()
        kmeans = KMeans(n_clusters=query_size)
        kmeans.fit(embeddings)

        cluster_preds = kmeans.predict(embeddings)
        centers = kmeans.cluster_centers_[cluster_preds]
        dis = (embeddings - centers) ** 2
        dis = dis.sum(axis=1)
        top_n = np.array(
            [
                np.arange(embeddings.shape[0])[cluster_preds == i][
                    dis[cluster_preds == i].argmin()
                ]
                for i in range(query_size)
            ]
        )

        return unlab_inds[top_n]


class AntiKMeansSampler(Sampler):
    name = "anti_kmeans"

    def query(self, query_size, unlab_inds, model, **kwargs):
        embeddings = self._forward_iter(unlab_inds, model.get_encoded).cpu().numpy()
        kmeans = KMeans(n_clusters=query_size)
        kmeans.fit(embeddings)

        cluster_preds = kmeans.predict(embeddings)
        centers = kmeans.cluster_centers_[cluster_preds]
        dis = (embeddings - centers) ** 2
        dis = dis.sum(axis=1)
        top_n = np.array(
            [
                np.arange(embeddings.shape[0])[cluster_preds == i][
                    dis[cluster_preds == i].argmax()
                ]
                for i in range(query_size)
            ]
        )

        return unlab_inds[top_n]
