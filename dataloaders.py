import os
import torch

import numpy as np
import pandas as pd
from functools import partial

from text import Field, LabelField, Iterator, BucketIterator, TabularDataset


from transformers import AutoTokenizer


class TokenizerVocabWrapper:
    def __init__(self, tokenizer):
        # Wrap BertTokenizer so the method signatures align with podium
        self.tokenizer = tokenizer

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.tokenizer)

    def numericalize(self, instance):
        # Equivalent to .encode, but I want to delineate the steps
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instance))


def make_iterable(
    dataset, device, batch_size=32, shuffle=False, indices=None, bucket=False
):
    """
    Construct a DataLoader from a podium Dataset
    """

    def instance_length(instance):
        raw, tokenized = instance.text
        return -len(tokenized)

    def cast_to_device(data):
        return torch.tensor(data, device=device)

    # Selects examples at given indices to support subset iteration.
    if indices is not None:
        dataset = dataset[indices]

    if bucket:
        iterator = BucketIterator(
            dataset,
            batch_size=batch_size,
            sort_key=instance_length,
            shuffle=shuffle,
            matrix_class=cast_to_device,
            look_ahead_multiplier=20,
        )
    else:
        iterator = Iterator(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            matrix_class=cast_to_device,
        )

    return iterator


class Instance:
    def __init__(self, index, text, label, extras=None):
        self.index = index
        self.text = text
        self.label = label
        self.extras = extras
        self.length = len(text)  # text is already tokenized & filtered

    def set_mask(self, masked_text, masked_labels):
        # Set the masking as an attribute
        self.masked_text = masked_text
        self.masked_labels = masked_labels

    def set_numericalized(self, indices, target):
        self.numericalized_text = indices
        self.numericalized_label = target
        self.length = len(indices)

    def __repr__(self):
        return f"{self.index}: {self.length}, {self.label}"


class MaxLenHook:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, raw, tokenized):
        return raw, tokenized[: self.max_len]


def lowercase_hook(raw, tokenized):
    return raw, [tok.lower() for tok in tokenized]


def isalnum(token):
    return any(c.isalnum() for c in token)


def remove_nonalnum(raw, tokenized):
    # Remove non alphanumeric tokens
    return raw, [tok for tok in tokenized if isalnum(tok)]


def load_imdb(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/IMDB",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_isear(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/ISEAR",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_agn2(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/AGN-2",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_agn4(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/AGN-4",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_mnli(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):
    return load_sequence_pair_dataset(
        "data/GLUE/MNLI",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_rte(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):
    return load_sequence_pair_dataset(
        "data/GLUE/RTE",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_mrpc(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):
    return load_sequence_pair_dataset(
        "data/GLUE/MRPC",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_qqp(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):
    return load_sequence_pair_dataset(
        "data/GLUE/QQP",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_qnli(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):
    return load_sequence_pair_dataset(
        "data/GLUE/QNLI",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_sst(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):
    return load_dataset(
        "data/GLUE/SST-2",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_trec2(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/TREC-2",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_trec6(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/TREC-6",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_cola(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=128,
):

    return load_dataset(
        "data/GLUE/COLA",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_polarity(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/POL",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_subj(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/SUBJ",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


# def test_load_cola(meta, tok):
#     splits, vocab = load_cola(meta, tok)
#     print(vocab)
#     train, valid, test = splits
#     print(len(train), len(valid), len(test))

#     print(train)
#     print(train[0])

#     device = torch.device("cpu")
#     train_iter = make_iterable(test, device, batch_size=2)
#     batch = next(iter(train_iter))

#     print(batch)
#     text, length = batch.text
#     print(length[0])
#     print(vocab.get_padding_index())


def load_sequence_pair_dataset(
    data_dir, meta, tokenizer=None, max_vocab_size=20_000, max_seq_len=200
):

    # Use BERT subword tokenization
    vocab = TokenizerVocabWrapper(tokenizer)
    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    fields = [
        Field("id", disable_batch_matrix=True),
        Field(
            "sequence1",
            tokenizer=tokenizer.tokenize,
            padding_token=pad_index,
            numericalizer=tokenizer.convert_tokens_to_ids,
            include_lengths=True,
            posttokenize_hooks=[
                MaxLenHook(max_seq_len),
                lowercase_hook,
            ],
        ),
        Field(
            "sequence2",
            tokenizer=tokenizer.tokenize,
            padding_token=pad_index,
            numericalizer=tokenizer.convert_tokens_to_ids,
            include_lengths=True,
            posttokenize_hooks=[
                MaxLenHook(max_seq_len),
                lowercase_hook,
            ],
        ),
        LabelField("label"),
    ]

    train = TabularDataset(
        os.path.join(data_dir, "train.csv"), format="csv", fields=fields
    )
    val = TabularDataset(
        os.path.join(data_dir, "validation.csv"), format="csv", fields=fields
    )
    test = TabularDataset(
        os.path.join(data_dir, "test.csv"), format="csv", fields=fields
    )

    train.finalize_fields()

    meta.vocab = vocab
    meta.num_tokens = len(vocab)
    meta.padding_idx = vocab.get_padding_index()
    meta.num_labels = len(train.field("label").vocab)

    return (train, val, test), vocab


def load_dataset(
    data_dir, meta, tokenizer=None, max_vocab_size=20_000, max_seq_len=200
):

    # Use BERT subword tokenization
    vocab = TokenizerVocabWrapper(tokenizer)
    pad_index = tokenizer.pad_token_id
    fields = [
        Field("id", disable_batch_matrix=True),
        Field(
            "text",
            tokenizer=tokenizer.tokenize,
            padding_token=pad_index,
            numericalizer=tokenizer.convert_tokens_to_ids,
            posttokenize_hooks=[
                MaxLenHook(max_seq_len),
                # lowercase_hook,
            ],
        ),
        LabelField("label"),
    ]

    train = TabularDataset(
        os.path.join(data_dir, "train.csv"), format="csv", fields=fields
    )
    val = TabularDataset(
        os.path.join(data_dir, "validation.csv"), format="csv", fields=fields
    )
    test = TabularDataset(
        os.path.join(data_dir, "test.csv"), format="csv", fields=fields
    )

    train.finalize_fields()

    meta.vocab = vocab
    meta.num_tokens = len(vocab)
    meta.padding_idx = pad_index
    meta.num_labels = len(train.field("label").vocab)

    return (train, val, test)


def add_ids_to_files(root_folder):
    split_ins = ["train_old.csv", "dev_old.csv", "test_old.csv"]
    split_outs = ["train.csv", "dev.csv", "test.csv"]

    for split_in, split_out in zip(split_ins, split_outs):
        with open(os.path.join(root_folder, split_in), "r") as infile:
            with open(os.path.join(root_folder, split_out), "w") as outfile:
                for idx, line in enumerate(infile):
                    parts = line.strip().split(",")
                    if idx == 0:
                        continue
                    outfile.write(f"{idx-1},{parts[0]},{parts[1]}\n")


if __name__ == "__main__":
    data_dir = "data/SUBJ"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = TokenizerVocabWrapper(tokenizer)
    fields = [
        Field("id", disable_batch_matrix=True),
        Field(
            name="text",
            tokenizer=tokenizer.tokenize,
            padding_token=vocab.get_pad_token_id(),
            numericalizer=tokenizer.convert_tokens_to_ids,
            posttokenize_hooks=[
                remove_nonalnum,
                MaxLenHook(200),
                lowercase_hook,
            ],
        ),
        LabelField("label"),
    ]

    train = TabularDataset(
        os.path.join(data_dir, "train.csv"), format="csv", fields=fields
    )
    val = TabularDataset(
        os.path.join(data_dir, "validation.csv"), format="csv", fields=fields
    )
    test = TabularDataset(
        os.path.join(data_dir, "test.csv"), format="csv", fields=fields
    )

    train.finalize_fields()

    print(train[0])

    iter_ = Iterator(train, batch_size=2)
    for batch in iter_:
        print(batch.id)
