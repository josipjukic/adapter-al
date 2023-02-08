from dataloaders import *

import argparse

import transformers


transformers.logging.set_verbosity_error()


dataset_loaders = {
    "IMDB": load_imdb,
    "POL": load_polarity,
    "TREC-2": load_trec2,
    "TREC-6": load_trec6,
    "COLA": load_cola,
    "SUBJ": load_subj,
    "SST": load_sst,
    "AGN-2": load_agn2,
    "AGN-4": load_agn4,
    "ISEAR": load_isear,
    "MNLI": load_mnli,
    "MRPC": load_mrpc,
    "QQP": load_qqp,
    "QNLI": load_qnli,
}

pair_sequence_datasets = {
    "MNLI",
    "QNLI",
    "MRPC",
    "QQP",
    "RTE",
}


unbiased_estimators = ["PURE", "LURE"]


def make_parser():
    parser = argparse.ArgumentParser(description="Active Learning")
    parser.add_argument(
        "--data",
        type=str,
        default="TREC-2",
        help="Data corpus.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BERT",
        help="Model: [ALBERT, BERT, ELECTRA, RoBERTa, DistilBERT]",
    )

    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="N", help="batch size"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
    parser.add_argument(
        "--l2", type=float, default=1e-5, help="l2 regularization (weight decay)"
    )
    parser.add_argument("--freeze", action="store_true", help="Freeze embeddings")

    # Vocab specific arguments
    parser.add_argument(
        "--max-vocab", type=int, default=10000, help="maximum size of vocabulary"
    )
    parser.add_argument(
        "--min-freq", type=int, default=5, help="minimum word frequency"
    )
    parser.add_argument(
        "--max_len", type=int, default=200, help="maximum length of input sequence"
    )

    # Repeat experiments
    parser.add_argument(
        "--repeat", type=int, default=5, help="number of times to repeat training"
    )

    # Gpu based arguments
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="Gpu to use for experiments (-1 means no GPU)",
    )

    # Storing & loading arguments
    parser.add_argument(
        "--save",
        type=str,
        default="chkp/",
        help="Folder to store final model",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Folder to store final model (or model with best valid perf) in",
    )
    parser.add_argument(
        "--log", type=str, default="tb_log/", help="Folder to store tensorboard logs in"
    )
    parser.add_argument(
        "--restore", type=str, default="", help="File to restore model from"
    )

    # Active learning arguments
    parser.add_argument(
        "--al-samplers",
        nargs="+",
        default=["random", "entropy", "entropy_dropout", "dal", "core_set"],
        choices=[
            "random",
            "entropy",
            "kmeans",
            "entropy_dropout",
            "core_set",
            "dal",
            "anti_entropy",
        ],
        help="Specify a list of active learning samplers.",
    )
    parser.add_argument(
        "--al-epochs",
        type=int,
        default=-1,
        help="Number of AL epochs (-1 uses the whole train set)",
    )
    parser.add_argument(
        "--query-size", type=int, default=50, help="Active learning query size."
    )
    parser.add_argument(
        "--warm-start-size", type=int, default=50, help="Initial AL batch size."
    )
    parser.add_argument(
        "--max-train-size", type=int, default=1000, help="Maximum train set size."
    )

    parser.add_argument(
        "--unbiased",
        type=str,
        default="none",
        help="Unbiased estimator: [PURE, LURE]. If 'none', then ignored.",
    )

    parser.add_argument(
        "--repr-stats",
        type=bool,
        default=False,
        help="Calculate representation space statistics.",
    )

    parser.add_argument(
        "--pretrain",
        type=str,
        choices=["standard", "adapter"],
        help="TAPT(A).",
    )

    parser.add_argument(
        "--besov",
        type=bool,
        default=False,
        help="Use Besov-optimal strategy.",
    )

    parser.add_argument(
        "--best",
        type=bool,
        default=False,
        help="Use best accuracy.",
    )

    parser.add_argument(
        "--stratified",
        type=bool,
        default=False,
        help="Stratified warm start sample.",
    )

    parser.add_argument(
        "--scheduler",
        type=bool,
        default=False,
        help="Use linear decay scheduler.",
    )

    parser.add_argument(
        "--adapter",
        type=str,
        help="Adapter.",
    )

    parser.add_argument(
        "--share-encoders", type=bool, default=True, help="Share encoders."
    )

    return parser.parse_args()
