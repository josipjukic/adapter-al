import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloaders import make_iterable

from ..sampler import Sampler
from .mc_dropout import ConsistentMCDropout
from .bald_utils import get_batchbald_batch


class BatchBALDDropout(Sampler):
    """
    Implementation of BatchBALD Strategy :footcite:`kirsch2019batchbald`, which refines
    the original BALD acquisition to the batch setting using a new acquisition function.
    This class extends :class:`active_learning_strategies.strategy.Strategy`
    to include a MC sampling technique based on the sampling techniques used in their paper.
    """

    name = "batch_bald"

    def __init__(
        self,
        dataset,
        batch_size,
        device,
        n_drop=30,
        n_samples=1000,
        mod_inject="decoder",
    ):
        self.n_drop = n_drop
        self.n_samples = n_samples
        self.mod_inject = mod_inject

        super().__init__(dataset, batch_size, device)

    def do_MC_dropout_before_linear(self, model, unlab_inds, num_targets):
        data = make_iterable(
            self.dataset,
            self.device,
            batch_size=self.batch_size,
            indices=unlab_inds,
        )

        # Check that there is a linear layer attribute in the supplied model
        try:
            getattr(model, self.mod_inject)
        except:
            raise ValueError(
                f"Model does not have attribute {self.mod_inject} as the last layer"
            )

        # Store the linear layer in a temporary variable
        lin_layer_temp = getattr(model, self.mod_inject)

        # Inject dropout into the model by using ConsistentMCDropout module from BatchBALD repo
        dropout_module = ConsistentMCDropout()
        dropout_injection = torch.nn.Sequential(dropout_module, lin_layer_temp)
        setattr(model, self.mod_inject, dropout_injection)

        # Make sure that the model is in eval mode
        model.eval()

        # For safety, explicitly set the dropout module to be in evaluation mode
        dropout_module.train(mode=False)

        # Create a tensor that will store the probabilities
        probs = torch.zeros([self.n_drop, len(unlab_inds), num_targets]).to(self.device)
        with torch.no_grad():
            for i in range(self.n_drop):
                evaluated_points = 0

                # In original BatchBALD code, inference samples were predicted in a single forward pass via an additional forward parameter.
                # Hence, only 1 mask needed to be generated during eval time for consistent MC sampling (as there was only 1 pass). Here,
                # our models do not assume this forward parameter. Hence, we must have a different generated mask for each PASS of the
                # dataset. Note, however, that the mask is CONSISTENT within a pass, which is functionally equivalent to the original
                # BatchBALD code.
                dropout_module.reset_mask()

                for batch in data:
                    x, lengths = batch.text
                    idxs = [
                        iter_index
                        for iter_index in range(
                            evaluated_points, evaluated_points + len(x)
                        )
                    ]
                    out, _ = model(x, lengths=lengths)
                    probs[i][idxs] = F.softmax(out, dim=1)
                    evaluated_points += len(x)

        # Transpose the probs to match BatchBALD interface
        probs = probs.permute(1, 0, 2)

        # Restore the linear layer
        setattr(model, self.mod_inject, lin_layer_temp)

        # Return the MC samples for each data instance
        return probs

    def query(self, query_size, unlab_inds, lab_inds, model, num_targets, **kwargs):
        # Get the MC samples.
        probs = self.do_MC_dropout_before_linear(model, unlab_inds, num_targets)

        # Compute the log probabilities to match BatchBALD interface.
        log_probs = torch.log(probs)

        # Use BatchBALD to select the new points.
        print("Query size: ", query_size)
        _, top_n = get_batchbald_batch(
            log_probs, query_size, self.n_samples, device=self.device
        )
        print("TOP N", len(top_n))

        return unlab_inds[top_n]
