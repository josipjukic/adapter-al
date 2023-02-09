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
    seeds = list(range(1, args.repeat + 1))

    dataloader = dataset_loaders[args.data]


    tokenizer = None
    if args.model in TRANSFORMERS.keys():
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS[args.model])

    meta = Config()

    (train, val, test), vocab = dataloader(meta=meta, tokenizer=tokenizer)

    if args.data in pair_sequence_datasets:
        meta.pair_sequence = True
    else:
        meta.pair_sequence = False

    for sampler_name in args.al_samplers:
        result_list = []

        # Initialize logging
        fmt = "%Y-%m-%d-%H-%M"
        start_time = fname = datetime.now().strftime(fmt)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    f"log/{args.data}-{args.model}-{sampler_name}-{start_time}.log"
                ),
                logging.StreamHandler(),
            ],
        )

        meta_info = {
            "dataset": args.data,
            "model": args.model,
            "al_sampler": sampler_name,
            "warm_start_size": args.warm_start_size,
            "query_size": args.query_size,
            "batch_size": args.batch_size,
            "epochs_per_train": args.epochs,
            "seeds": seeds,
            "lr": args.lr,
            "l2": args.l2,
        }

        logging.info(meta_info)

        for i, seed in zip(range(1, args.repeat + 1), seeds):
            logging.info(f"Running experiment {i}/{args.repeat}")
            logging.info(f"=" * 100)

            set_seed_everywhere(seed)
            logging.info(f"Seed = {seed}")
            logging.info(f"Maximum train size: {len(train)}")

            meta.num_labels = len(train.field("label").vocab)

            cuda = torch.cuda.is_available() and args.gpu != -1
            device = (
                torch.device("cpu") if not cuda else torch.device(f"cuda:{args.gpu}")
            )

            # Setup the loss function
            if meta.num_labels == 2:
                # Binary classification
                criterion = nn.BCEWithLogitsLoss()
                meta.num_targets = 1
            else:
                # Multiclass classification
                criterion = nn.CrossEntropyLoss()
                meta.num_targets = meta.num_labels

            sampler_cls = get_al_sampler(sampler_name)
            sampler = sampler_cls(
                dataset=train,
                batch_size=args.batch_size,
                device=device,
                meta=meta,
                tokenizer=tokenizer,
            )
            active_learner = Experiment(sampler, train, test, device, args, meta)

            results = active_learner.al_loop(
                create_model_fn=initialize_model,
                criterion=criterion,
                warm_start_size=args.warm_start_size,
                query_size=args.query_size,
                tokenizer=tokenizer,
            )

            result_list.append(results)

        wandb.config = meta_info
        fname = (
            f"{args.data}-{args.model}-{args.adapter}-{sampler.name}-"
            f"lm={args.pretrain}-e={args.epochs}-{start_time}.pkl"
        )
        meta_info["time"] = (
            datetime.now() - datetime.strptime(start_time, fmt)
        ).total_seconds()

        with open(f"{args.save_dir}/{fname}", "wb") as f:
            pickle.dump((result_list, meta_info), f)
