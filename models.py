from abc import abstractmethod
from functools import partial

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoAdapterModel,
)

from transformers.adapters import (
    PfeifferConfig,
    HoulsbyConfig,
    CompacterConfig,
    PrefixTuningConfig,
    LoRAConfig,
    IA3Config,
    MAMConfig,
    UniPELTConfig,
    ParallelConfig,
)


# SequenceClassifierOutput = namedtuple(
#     "SequenceClassifierOutput", ["loss", "logits", "hidden_states", "attentions"]
# )


class AcquisitionModel:
    @abstractmethod
    def get_encoder_dim(self, **kwargs):
        pass

    @abstractmethod
    def get_encoder(self, **kwargs):
        pass

    @abstractmethod
    def predict_probs(self, **kwargs):
        pass


class Transformer(nn.Module, AcquisitionModel):
    def __init__(self, config, meta, name, pretrained=None, adapter=False):
        super().__init__()

        self.name = name

        if pretrained == "standard":
            name = f"pretrained/{config.model}-{config.data}"
        else:
            name = TRANSFORMERS[name]

        if adapter:
            self.classifier = AutoAdapterModel.from_pretrained(name)
            if pretrained == "adapter":
                task_name = f"{config.data}-{config.model}-{config.adapter}"
                self.classifier.add_classification_head(
                    task_name, num_labels=meta.num_targets
                )
                self.classifier.load_adapter(f"adapters/{task_name}", with_head=False)
            else:
                task_name = config.data
                self.classifier.add_classification_head(
                    task_name, num_labels=meta.num_targets
                )
                adapter_config = ADAPTER_CONFIGS[adapter]()
                self.classifier.add_adapter(task_name, config=adapter_config)
            # Enable adapter training
            self.classifier.train_adapter(task_name)
        else:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                name, num_labels=meta.num_targets
            )
        self.num_targets = meta.num_targets

    def forward(self, inputs, lengths=None):
        output = self.classifier(inputs, output_hidden_states=True)
        logits = output.logits
        hidden_states = output.hidden_states
        e = hidden_states[0].mean(dim=1)
        hidden = hidden_states[-1][:, 0, :]
        return_dict = {
            "embeddings": e,
            "encoded": hidden,
        }

        return logits, return_dict

    def predict_probs(self, inputs, lengths=None):
        with torch.inference_mode():
            logits, _ = self(inputs, lengths)
            if self.num_targets == 1:
                # Binary classification
                y_pred = torch.sigmoid(logits)
                y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
            else:
                # Multiclass classification
                y_pred = F.softmax(logits, dim=1)
            return y_pred

    def get_encoder_dim(self):
        return self.classifier.config.hidden_size

    def get_encoded(self, inputs, lengths=None):
        with torch.inference_mode():
            output = self.classifier(inputs, output_hidden_states=True)
            hidden = output.hidden_states[-1][:, 0, :]
            return hidden

    def get_classifier_name(self):
        return TRANSFORMER_CLASSIFIERS[self.name]


class Transformer2(nn.Module, AcquisitionModel):
    def __init__(self, config, meta, name, pretrained=None, adapter=False):
        super().__init__()

        self.name = name

        model_cls = MODEL_CLS[meta.task_type]

        if pretrained == "standard":
            name = f"pretrained/{config.model}-{config.data}"
        else:
            name = TRANSFORMERS[name]

        if adapter:
            task_name = f"{config.data}-{config.model}-{config.adapter}"
            self.classifier = model_cls.from_pretrained(
                name, num_labels=meta.num_targets
            )
            if pretrained == "adapter":
                self.classifier.load_adapter(f"adapters/{task_name}", with_head=False)
            else:
                adapter_config = ADAPTER_CONFIGS[adapter]()
                self.classifier.add_adapter(task_name, config=adapter_config)
            # Enable adapter training
            self.classifier.train_adapter(task_name)
        else:
            self.classifier = model_cls.from_pretrained(
                name, num_labels=meta.num_targets
            )
        self.num_targets = meta.num_targets

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        output = self.classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        loss = output.loss
        logits = output.logits
        hidden_states = output.hidden_states
        e = hidden_states[0].mean(dim=1)
        hidden = hidden_states[-1][:, 0, :]
        return_dict = {"embeddings": e, "encoded": hidden, "loss": loss}

        return logits, return_dict

    def predict_probs(self, inputs, lengths=None):
        with torch.inference_mode():
            logits, _ = self(inputs, lengths)
            if self.num_targets == 1:
                # Binary classification
                y_pred = torch.sigmoid(logits)
                y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
            else:
                # Multiclass classification
                y_pred = F.softmax(logits, dim=1)
            return y_pred

    def get_encoder_dim(self):
        return self.classifier.config.hidden_size

    def get_encoded(self, inputs, lengths=None):
        with torch.inference_mode():
            output = self.classifier(inputs, output_hidden_states=True)
            hidden = output.hidden_states[-1][:, 0, :]
            return hidden

    def get_classifier_name(self):
        return TRANSFORMER_CLASSIFIERS[self.name]


def initialize_model(args, meta, pretrained=None):
    model_cls = models[args.model]
    model = model_cls(
        config=args, meta=meta, pretrained=pretrained, adapter=args.adapter
    )

    return model


# class PairSequenceClassifier(nn.Module, AcquisitionModel):
#     def __init__(self, config, meta, name, pretrained=None, adapter=False):
#         super().__init__()

#         if pretrained == "standard":
#             name = f"pretrained/{config.model}-{config.data}"
#         else:
#             name = TRANSFORMERS[name]

#         if adapter:
#             self.classifier = AutoAdapterModel.from_pretrained(name)
#             if pretrained == "adapter":
#                 task_name = f"{config.data}-{config.model}-{config.adapter}"
#                 self.classifier.add_classification_head(
#                     task_name, num_labels=meta.num_targets
#                 )
#                 self.classifier.load_adapter(f"adapters/{task_name}", with_head=False)
#             else:
#                 task_name = config.data
#                 self.classifier.add_classification_head(
#                     task_name, num_labels=meta.num_targets
#                 )
#                 adapter_config = ADAPTER_CONFIGS[adapter]()
#                 self.classifier.add_adapter(task_name, config=adapter_config)
#             # Enable adapter training
#             self.classifier.train_adapter(task_name)
#         else:
#             self.classifier = AutoModelForSequenceClassification.from_pretrained(
#                 name, num_labels=meta.num_targets
#             )
#         self.num_targets = meta.num_targets

#         self.first_sequence_encoder = AutoModel.from_pretrained(name)
#         if config.share_encoders:
#             # Use same encoder for both sequences
#             self.second_sequence_encoder = self.first_sequence_encoder
#         else:
#             # New set of parameters
#             self.second_sequence_encoder = AutoModel.from_pretrained(name)

#         self.num_targets = meta.num_targets
#         self.model_config = (
#             self.first_sequence_encoder.config
#         )  # Fetch for hidden state size
#         self.encoder_hidden = self.model_config.hidden_size

#         self.decoder = nn.Linear(
#             2 * self.encoder_hidden, self.num_targets
#         )  # concat > decoder

#     def encode(self, first_sequence, second_sequence, lengths=None):
#         encoded_first_sequence = self.first_sequence_encoder(
#             first_sequence, output_hidden_states=True
#         ).hidden_states
#         embedded_first_sequence = encoded_first_sequence[0]
#         encoded_first_sequence = encoded_first_sequence[-1].mean(
#             dim=1
#         )  # Average across T

#         encoded_second_sequence = self.second_sequence_encoder(
#             second_sequence, output_hidden_states=True
#         ).hidden_states
#         embedded_second_sequence = encoded_second_sequence[0]
#         encoded_second_sequence = encoded_second_sequence[-1].mean(
#             dim=1
#         )  # Average across T
#         return (
#             encoded_first_sequence,
#             encoded_second_sequence,
#             embedded_first_sequence,
#             embedded_second_sequence,
#         )

#     def forward(self, first_sequence, second_sequence, lengths=None):

#         h_1, h_2, e_1, e_2 = self.encode(first_sequence, second_sequence, lengths)

#         pair_sequence_encoding = torch.cat([h_1, h_2], dim=-1)  # Concatenate along [h]

#         output = self.decoder(pair_sequence_encoding)
#         return_dict = {
#             "embeddings": (e_1, e_2),
#             "encoded": (h_1, h_2),
#         }

#         return output, return_dict

#     def predict_probs(self, first_sequence, second_sequence, lengths=None):
#         with torch.inference_mode():
#             logits, _ = self(first_sequence, second_sequence, lengths)
#             if self.num_targets == 1:
#                 # Binary classification
#                 y_pred = torch.sigmoid(logits)
#                 y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
#             else:
#                 # Multiclass classification
#                 y_pred = F.softmax(logits, dim=1)
#             return y_pred

#     def get_encoder_dim(self):
#         return self.model_config.hidden_size * 2

#     def get_encoded(self, first_sequence, second_sequence, lengths=None):
#         with torch.inference_mode():
#             h_1, h_2, _, _ = self.encode(first_sequence, second_sequence, lengths)
#             return torch.cat([h_1, h_2], dim=-1)


def initialize_language_model(args, meta):
    model_cls = TRANSFORMERS[args.model]
    lm = AutoModelForMaskedLM.from_pretrained(model_cls)
    if args.adapter:
        adapter_config = ADAPTER_CONFIGS[args.adapter]()
        task_name = f"{args.data}-{args.model}-{args.adapter}"
        lm.add_adapter(
            task_name,
            config=adapter_config,
        )
        lm.train_adapter(task_name)
    return lm


TRANSFORMERS = {
    "BERT": "bert-base-uncased",
    "ALBERT": "albert-base-v2",
    "ELECTRA": "google/electra-base-discriminator",
    "DistilBERT": "distilbert-base-uncased",
    "RoBERTa": "roberta-base",
}

TRANSFORMER_CLASSIFIERS = {
    "BERT": "bert",
    "ALBERT": "albert",
    "ELECTRA": "electra",
    "DistilBERT": "distilbert",
    "RoBERTa": "roberta",
}


models = {
    "BERT": partial(Transformer2, name="BERT"),
    "ALBERT": partial(Transformer, name="ALBERT"),
    "ELECTRA": partial(Transformer, name="ELECTRA"),
    "DistilBERT": partial(Transformer, name="DistilBERT"),
    "RoBERTa": partial(Transformer, name="RoBERTa"),
}


ADAPTER_CONFIGS = {
    "pfeiffer": PfeifferConfig,
    "houlsby": HoulsbyConfig,
    "prefix": PrefixTuningConfig,
    "parallel": ParallelConfig,
    "lora": LoRAConfig,
    "ia3": IA3Config,
    "mam": MAMConfig,
    "compacter": CompacterConfig,
    "unipelt": UniPELTConfig,
}


MODEL_CLS = {
    "clf": AutoModelForSequenceClassification,
    "seq": AutoModelForTokenClassification,
}
