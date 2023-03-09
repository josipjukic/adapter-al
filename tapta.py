from al.experiment import Experiment
from dataloaders import *
from args import *

import logging
from datetime import datetime
from models import *
from util import set_seed_everywhere, Config


from transformers import AutoTokenizer


if __name__ == "__main__":
    args = make_parser()

    dataloader = dataset_loaders[args.data]

    meta = Config()

    tokenizer = None
    if args.model in TRANSFORMERS.keys():
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS[args.model])

    (train, val, test) = dataloader(meta=meta, tokenizer=tokenizer)

    if args.data in pair_sequence_datasets:
        meta.pair_sequence = True
    else:
        meta.pair_sequence = False

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

    # test_lengths = [len(ex.text[1]) for ex in test.examples]
    meta_info = {
        "dataset": args.data,
        "model": args.model,
        "batch_size": args.batch_size,
        "l2": args.l2,
    }
    logging.info(meta_info)

    logging.info(f"Running experiment")
    logging.info(f"=" * 100)

    seed = 0
    set_seed_everywhere(seed)
    logging.info(f"Seed = {seed}")
    logging.info(f"Maximum train size: {len(train)}")

    cuda = torch.cuda.is_available() and args.gpu != -1
    device = torch.device("cpu") if not cuda else torch.device(f"cuda:{args.gpu}")

    # Setup the loss fn
    if meta.num_labels == 2:
        # Binary classification
        criterion = nn.BCEWithLogitsLoss()
        meta.num_targets = 1
    else:
        # Multiclass classification
        criterion = nn.CrossEntropyLoss()
        meta.num_targets = meta.num_labels

    experiment = Experiment(None, train, test, device, args, meta)
    # meta_info["test_lengths"] = experiment.test_lengths.tolist()
    # meta_info["test_mapping"] = experiment.get_test_id_mapping()

    mlm = experiment._pretrain_lm(tokenizer, epochs=args.epochs)
    mlm.save_adapter(
        f"adapters/{args.data}-{args.model}-{args.adapter}",
        f"{args.data}-{args.model}-{args.adapter}",
        with_head=False,
    )
