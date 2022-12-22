import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os
import random


class Config(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError

    def __setattr__(self, key, value):
        self[key] = value


def create_pad_mask_from_length(tensor, lengths):
    # Creates a mask where `True` is on the non-padded locations
    # and `False` on the padded locations
    mask = torch.arange(tensor.size(-1))[None, :].to(lengths.device) < lengths[:, None]
    mask = mask.to(tensor.device)
    return mask


def generate_permutation(batch, lengths):
    # batch contains attention weights post scaling
    # [BxT]
    batch_size, max_len = batch.shape
    # repeat arange for batch_size times
    perm_idx = np.tile(np.arange(max_len), (batch_size, 1))

    for batch_index, length in enumerate(lengths):
        perm = np.random.permutation(length.item())
        perm_idx[batch_index, :length] = perm

    return torch.tensor(perm_idx)


def replace_with_uniform(tensor, lengths):
    # Assumed: [BxT] shape for tensor
    uniform = create_pad_mask_from_length(tensor, lengths).type(torch.float)
    for idx, l in enumerate(lengths):
        uniform[idx] /= l
    return uniform


def masked_softmax(attn_odds, masks):
    attn_odds.masked_fill_(~masks, -float("inf"))
    attn = F.softmax(attn_odds, dim=-1)
    return attn


def create_pad_mask_from_length(tensor, lengths, idx=-1):
    # Creates a mask where `True` is on the non-padded locations
    # and `False` on the padded locations
    mask = torch.arange(tensor.size(idx))[None, :].to(lengths.device) < lengths[:, None]
    mask = mask.to(tensor.device)
    return mask


def set_seed_everywhere(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logits_to_probs(logits):
    num_targets = logits.shape[-1]
    if num_targets == 1:
        # Binary classification
        y_pred = torch.sigmoid(logits)
        y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
    else:
        # Multiclass classification
        y_pred = F.softmax(logits, dim=1)
    return y_pred


def compute_forgetfulness(epochwise_tensor):
    """
    Given a epoch-wise trend of train predictions, compute frequency with which
    an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
    Based on: https://arxiv.org/abs/1812.05159
    """

    out = []

    datawise_tensor = epochwise_tensor.transpose(0, 1)
    for correctness_trend in datawise_tensor:
        if not any(
            correctness_trend
        ):  # Example is never predicted correctly, or learnt!
            out.append(torch.tensor(1000))
            continue
        learnt = False  # Predicted correctly in the current epoch.
        times_forgotten = 0
        for is_correct in correctness_trend:
            if (not learnt and not is_correct) or (learnt and is_correct):
                # Nothing changed.
                continue
            elif learnt and not is_correct:
                # Forgot after learning at some point!
                learnt = False
                times_forgotten += 1
            elif not learnt and is_correct:
                # Learnt!
                learnt = True
        out.append(torch.tensor(times_forgotten))

    return torch.stack(out)


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
