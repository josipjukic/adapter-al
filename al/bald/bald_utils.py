import torch
import math

from . import joint_entropy


def compute_conditional_entropy(log_probs):
    N, K, _ = log_probs.shape

    entropies_N = torch.empty(N, dtype=torch.double)
    nats_N_K_C = log_probs * torch.exp(log_probs)
    entropies_N.copy_(-torch.sum(nats_N_K_C, dim=(1, 2)) / K)

    return entropies_N


def compute_entropy(log_probs):
    N, K, _ = log_probs.shape

    entropies_N = torch.empty(N, dtype=torch.double)
    mean_log_probs_N_C = torch.logsumexp(log_probs, dim=1) - math.log(K)
    nats_N_C = mean_log_probs_N_C * torch.exp(mean_log_probs_N_C)
    entropies_N.copy_(-torch.sum(nats_N_C, dim=1))

    return entropies_N


def get_batchbald_batch(
    log_probs,
    batch_size,
    num_samples,
    dtype=None,
    device=None,
):
    N, K, C = log_probs.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return (candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs)

    batch_joint_entropy = joint_entropy.DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in range(batch_size):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(
                log_probs[latest_index : latest_index + 1]
            )

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return (candidate_scores, candidate_indices)


def get_bald_batch(log_probs, batch_size):
    N, _, _ = log_probs.shape

    batch_size = min(batch_size, N)

    candidate_indices = []

    scores_N = -compute_conditional_entropy(log_probs)
    scores_N += compute_entropy(log_probs)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return (candiate_scores.tolist(), candidate_indices.tolist())
