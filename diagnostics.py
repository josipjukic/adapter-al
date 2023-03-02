import wandb
from al.sampler_mapping import get_al_sampler
from al.experiment import Experiment
from dataloaders import *

import pickle
import logging
from datetime import datetime
from transformers import logging as trans_log

from util import set_seed_everywhere

import torch
from transformers import AutoTokenizer

from util import Config
from dataloaders import *
from models import *

from args import *


if __name__ == "__main__":
    wandb.init(project="active-learning", entity="jjukic")
    trans_log.set_verbosity_error()
    args = make_parser()
    seed = 0

    dataloader = dataset_loaders[args.data]

    tokenizer = None
    if args.model in TRANSFORMERS.keys():
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS[args.model])

    meta = Config()

    (train, val, test) = dataloader(meta=meta, tokenizer=tokenizer)

    if args.data in pair_sequence_datasets:
        meta.pair_sequence = True
    else:
        meta.pair_sequence = False

    if args.data in seq_lab_datasets:
        meta.task_type = "seq"
    else:
        meta.task_type = "clf"

    result_list = []

    # Initialize logging
    fmt = "%Y-%m-%d-%H-%M"
    start_time = fname = datetime.now().strftime(fmt)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"log/{args.data}-{args.model}-{start_time}.log"),
            logging.StreamHandler(),
        ],
    )

    meta_info = {
        "dataset": args.data,
        "model": args.model,
        "warm_start_size": args.warm_start_size,
        "query_size": args.query_size,
        "batch_size": args.batch_size,
        "epochs_per_train": args.epochs,
        "seeds": seed,
        "lr": args.lr,
        "l2": args.l2,
    }

    logging.info(meta_info)

    logging.info(f"Running diagnostics")
    logging.info(f"=" * 100)

    set_seed_everywhere(seed)
    logging.info(f"Seed = {seed}")

    meta.num_labels = len(train.field("label").vocab)

    cuda = torch.cuda.is_available() and args.gpu != -1
    device = torch.device("cpu") if not cuda else torch.device(f"cuda:{args.gpu}")

    # Setup the loss function
    if meta.num_labels == 2:
        # Binary classification
        criterion = nn.BCEWithLogitsLoss()
        meta.num_targets = 1
    else:
        # Multiclass classification
        criterion = nn.CrossEntropyLoss()
        meta.num_targets = meta.num_labels

    experiment = Experiment(None, train, test, device, args, meta)

    df = experiment.cartography(
        create_model_fn=initialize_model, criterion=criterion, tokenizer=tokenizer
    )
    pvi_result = experiment.calculate_predictive_entropy(
        create_model_fn=initialize_model, criterion=criterion, tokenizer=tokenizer
    )
    print(pvi_result)
    df["pvi"] = pvi_result

    df.to_csv(f"stats/{args.model}-{args.data}-{args.adapter}.csv")
