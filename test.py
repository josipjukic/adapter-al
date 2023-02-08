from models import *
from args import *

args = Config()
args.lr = 2e-5
args.l2 = 0
args.model = "BERT"
args.data = "COLA"
args.adapter = None
args.batch_size = 32
args.epochs = 10
args.clip = 1

meta = Config()

dataloader = dataset_loaders[args.data]
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS[args.model])
(train, val, test), vocab = dataloader(meta=meta, tokenizer=tokenizer)

if args.data in pair_sequence_datasets:
    meta.pair_sequence = True
else:
    meta.pair_sequence = False

if meta.num_labels == 2:
    # Binary classification
    criterion = nn.BCEWithLogitsLoss()
    meta.num_targets = 1
else:
    # Multiclass classification
    criterion = nn.CrossEntropyLoss()
    meta.num_targets = meta.num_labels
    
model = Transformer(args, meta, args.model)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
device = torch.device("cpu")


def make_iterable(dataset, device, batch_size=32, train=False, indices=None):
    """
    Construct a DataLoader from a podium Dataset
    """

    def instance_length(instance):
        raw, tokenized = instance.text
        return -len(tokenized)

    def cast_to_device(data):
        return torch.tensor(np.array(data), device=device)

    # Selects examples at given indices to support subset iteration.
    if indices is not None:
        dataset = dataset[indices]

    iterator = BucketIterator(
        dataset,
        batch_size=batch_size,
        sort_key=instance_length,
        shuffle=train,
        matrix_class=cast_to_device,
        look_ahead_multiplier=20,
    )

    return iterator


model.to(device)
train_iter = make_iterable(
    test,
    device,
    batch_size=args.batch_size,
    train=True,
#     indices=indices,
)

for i in train_iter:
    print(i)
    break