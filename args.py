from dataloaders import *

import argparse
import os

import transformers

transformers.logging.set_verbosity_error()


word_vector_files = {"glove": os.path.expanduser("~/data/vectors/glove.840B.300d.txt")}


dataset_loaders = {
    "IMDB": load_imdb,
    "IMB": load_imb,
    "POL": load_polarity,
    "TREC": load_trec,
    "COLA": load_cola,
    "SUBJ": load_subj,
    "SST": load_sst,
    "ag_news": load_ag_news,
    "ag_news-full": load_ag_news_full,
    "TREC-full": load_trec_full,
    "ISEAR": load_isear,
}

pair_sequence_datasets = {
    "SNLI",
}


unbiased_estimators = ["PURE", "LURE"]


def make_parser():
    parser = argparse.ArgumentParser(description="Active Learning")
    parser.add_argument(
        "--data",
        type=str,
        default="IMDB",
        help="Data corpus: [IMDB, TREC, COLA]",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ELECTRA",
        help="Model: [JWA, MLP, ALBERT, BERT, ELECTRA, RoBERTa, DistilBERT, LR]",
    )

    # JWA arguments
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="LSTM",
        help="type of recurrent net [LSTM, GRU, MHA]",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="nqadd",
        help="attention type [dot, add, nqdot, nqadd], default = nqadd",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=300,
        help="size of word embeddings [Uses pretrained on 300]",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=150,
        help="number of hidden units for the encoder",
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="number of layers of the encoder"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument(
        "--vectors",
        type=str,
        default="glove",
        help="Pretrained vectors to use [glove, fasttext]",
    )
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=5, help="upper epoch limit")
    parser.add_argument(
        "--batch-size", type=int, default=32, metavar="N", help="batch size"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
    parser.add_argument(
        "--l2", type=float, default=1e-5, help="l2 regularization (weight decay)"
    )
    parser.add_argument("--bi", action="store_true", help="[USE] bidirectional encoder")
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
        default=["random", "entropy"],
        choices=[
            "random",
            "least_confident",
            "margin",
            "entropy",
            "kmeans",
            "least_confident_dropout",
            "margin_dropout",
            "entropy_dropout",
            "badge",
            "core_set",
            "batch_bald",
            "most_confident",
            "anti_margin",
            "anti_entropy",
            "anti_kmeans",
            "anti_badge",
            "anti_core_set",
            "entropy_sklearn",
            "margin_sklearn",
            "anti_entropy_sklearn",
            "dal",
            "repr",
            "anti_repr",
            "mean_repr",
            "anti_mean_repr",
            "albi",
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
        "--max-train-size", type=int, default=3000, help="Maximum train set size."
    )

    parser.add_argument(
        "--unbiased",
        type=str,
        default="none",
        help="Unbiased estimator: [PURE, LURE]. If 'none', then ignored.",
    )

    # Category arguments
    # ====================================

    # Category dataframe
    parser.add_argument("--category-df", type=str, help="Category data frame path.")

    # Load cartography
    parser.add_argument(
        "--load-cartography",
        type=str,
        help="Path to cartography data frame.",
    )

    # Easy proportion
    parser.add_argument(
        "--prop-easy",
        type=float,
        default=1.0,
        help="Proportion of easy examples for training in interval [0,1]."
        " E.g., 0.5 means that 50% of easy examples will be used for training.",
    )

    # Ambiguous proportion
    parser.add_argument(
        "--prop-amb",
        type=float,
        default=1.0,
        help="Proportion of ambiguous examples for training in interval [0,1]."
        " E.g., 0.5 means that 50% of ambiguous examples will be used for training.",
    )

    # Hard proportion
    parser.add_argument(
        "--prop-hard",
        type=float,
        default=1.0,
        help="Proportion of hard examples for training in interval [0,1]."
        " E.g., 0.5 means that 50% of hard examples will be used for training.",
    )

    # Load PVI
    parser.add_argument(
        "--load-pvi",
        type=str,
        help="Path to cartography data frame.",
    )

    parser.add_argument(
        "--pvi-threshold",
        type=float,
        help="PVI threshold. All of the examples below the threshold will be ignored.",
    )

    parser.add_argument(
        "--ex-loss",
        type=bool,
        default=False,
        help="Calculate expected loss.",
    )

    parser.add_argument(
        "--repr-stats",
        type=bool,
        default=False,
        help="Calculate representation space statistics.",
    )

    parser.add_argument(
        "--pretrain",
        type=bool,
        default=False,
        help="Include MLM pretraining in each AL step.",
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
        "--loss",
        type=bool,
        default=False,
        help="Use lowest loss.",
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
        type=bool,
        default=False,
        help="Adapter.",
    )

    return parser.parse_args()
