from pyrsistent import b
import numpy as np
from sklearn.linear_model import LogisticRegression as LR


from .sampler import Sampler


class DiscriminativeRepresentationSampling(Sampler):
    """
    An implementation of DAL (discriminative active learning), using the learned representation as our representation.
    This implementation is the one which performs best in practice.
    """

    name = "dal"

    def __init__(self, dataset, batch_size, device, meta, tokenizer):
        super().__init__(dataset, batch_size, device, meta, tokenizer)
        self.sub_batches = 10

    def query(self, query_size, unlab_inds, lab_inds, model, **kwargs):
        unlab_idx = np.random.choice(
            unlab_inds, np.min([lab_inds.size * 10, unlab_inds.size]), replace=False
        )
        embeddings = self._forward_iter(None, model.get_encoded)

        # iteratively sub-sample using the discriminative sampling routine:
        labeled_so_far = 0
        sub_sample_size = int(query_size / self.sub_batches)
        selected = []
        while labeled_so_far < query_size:
            if labeled_so_far + sub_sample_size > query_size:
                sub_sample_size = query_size - labeled_so_far

            clf = train_discriminative_model(
                embeddings,
                lab_inds,
                unlab_inds,
            )
            predictions = clf.predict_proba(embeddings[unlab_idx].cpu().numpy())[:, 1]
            top_n = np.argpartition(predictions, -sub_sample_size)[-sub_sample_size:]
            selected.extend(unlab_idx[top_n])
            lab_inds = np.concatenate([lab_inds, unlab_idx[top_n]])
            labeled_so_far += sub_sample_size
            unlab_inds = np.setdiff1d(unlab_inds, lab_inds)
            unlab_idx = np.random.choice(
                unlab_inds, np.min([lab_inds.size * 10, unlab_inds.size]), replace=False
            )

        return selected


def train_discriminative_model(embeddings, lab_inds, unlab_inds):
    """
    A function that trains and returns a discriminative model on the labeled and unlabeled data.
    """

    labeled = embeddings[lab_inds]
    unlabeled = embeddings[unlab_inds]

    # Create binary dataset
    y_L = np.zeros((labeled.shape[0], 1), dtype="int")
    y_U = np.ones((unlabeled.shape[0], 1), dtype="int")
    X_train = np.vstack((labeled.cpu().numpy(), unlabeled.cpu().numpy()))
    y_train = np.concatenate([y_L, y_U])

    # TODO: replace with MLP
    model = LR(solver="lbfgs", max_iter=10_000)
    model.fit(X_train, y_train.ravel())

    return model
