import numpy as np

from al.sampler import Sampler


class Representation(Sampler):

    name = "repr"

    def query(self, query_size, unlab_inds, model, device, **kwargs):
        repr_grad = (
            self._get_representation_gradient(
                model=model,
                indices=unlab_inds,
                device=device,
            )
            .detach()
            .cpu()
            .numpy()
        )

        top_n = np.argpartition(repr_grad, -query_size)[-query_size:]
        return unlab_inds[top_n]


class AntiRepresentation(Sampler):

    name = "anti_repr"

    def query(self, query_size, unlab_inds, model, device, **kwargs):
        repr_grad = (
            self._get_representation_gradient(
                model=model,
                indices=unlab_inds,
                device=device,
            )
            .detach()
            .cpu()
            .numpy()
        )

        print("REPR shape: ", repr_grad.shape)

        top_n = np.argpartition(-repr_grad, -query_size)[-query_size:]
        return unlab_inds[top_n]


class MeanRepresentation(Sampler):

    name = "mean_repr"

    def query(self, query_size, unlab_inds, model, device, **kwargs):
        repr_mean = (
            self._get_representation_mean(
                model=model,
                indices=unlab_inds,
                device=device,
            )
            .detach()
            .cpu()
            .numpy()
        )

        top_n = np.argpartition(repr_mean, -query_size)[-query_size:]
        return unlab_inds[top_n]


class AntiMeanRepresentation(Sampler):

    name = "anti_mean_repr"

    def query(self, query_size, unlab_inds, model, device, **kwargs):
        repr_mean = (
            self._get_representation_mean(
                model=model,
                indices=unlab_inds,
                device=device,
            )
            .detach()
            .cpu()
            .numpy()
        )

        top_n = np.argpartition(-repr_mean, -query_size)[-query_size:]
        return unlab_inds[top_n]
