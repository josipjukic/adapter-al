from email.parser import BytesParser
import numpy as np
import torch

from al.sampler import Sampler
from dataloaders import make_iterable
from smoothness.random_forest import WaveletsForestRegressor


class ActiveLearningBesovIndex(Sampler):

    name = "albi"

    def query(self, query_size, lab_inds, unlab_inds, model, device, **kwargs):

        lab_iter = make_iterable(
            self.dataset,
            self.device,
            batch_size=32,
            train=True,
            indices=lab_inds,
        )
        unlab_iter = make_iterable(
            self.dataset,
            self.device,
            batch_size=32,
            train=False,
            indices=unlab_inds,
        )

        print(f"NUM unlab: {len(unlab_inds)}")

        labels = []
        predicted_labels = []
        enc = []

        name = model.get_classifier_name()
        clf = getattr(model.classifier, name)
        config = model.classifier.config
        num_layers = config.num_hidden_layers

        enc_layers = {i: [] for i in range(num_layers)}
        enc_layers_unlab = {i: [] for i in range(num_layers)}
        results = {}

        with torch.no_grad():
            for batch in lab_iter:

                inputs, _ = batch.text
                labels.append(batch.label)

                # pad_idx = self.meta.padding_idx
                # attention_mask = inputs != pad_idx

                embedded_tokens = clf.embeddings(inputs)
                # head_mask = attention_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
                # head_mask = head_mask.expand(
                #     config.num_hidden_layers,
                #     -1,
                #     config.num_attention_heads,
                #     -1,
                #     attention_mask.shape[1],
                # )
                encoded_all = clf.encoder(
                    embedded_tokens,
                    output_hidden_states=True,
                    # head_mask=head_mask,
                    # attention_mask=attention_mask,
                )
                # Skip the embedding layer [1:]
                for i, enc_layer in enumerate(encoded_all[1][1:]):
                    enc_layers[i].append(enc_layer[:, 0].cpu())

        with torch.no_grad():
            for batch in unlab_iter:

                inputs, lengths = batch.text

                embedded_tokens = clf.embeddings(inputs)
                logits, _ = model(inputs, lengths)
                if logits.shape[-1] == 1:
                    pred = torch.ge(logits, 0).type(torch.long).squeeze()
                else:
                    _, pred = torch.max(logits, 1)

                if pred.dim() == 0:
                    pred = pred.reshape(1)
                predicted_labels.append(pred)

                encoded_all = clf.encoder(
                    embedded_tokens,
                    output_hidden_states=True,
                    # head_mask=head_mask,
                    # attention_mask=attention_mask,
                )
                # Skip the embedding layer [1:]
                for i, enc_layer in enumerate(encoded_all[1][1:]):
                    enc_layers_unlab[i].append(enc_layer[:, 0].cpu())

        for k, v in enc_layers.items():
            X = torch.cat(v).detach().numpy()
            y = torch.cat(labels).detach().cpu().numpy().ravel()
            n_values = np.max(y) + 1
            y = np.eye(n_values)[y]
            enc_layers[k] = (X, y)

        for k, v in enc_layers_unlab.items():
            X = torch.cat(v).detach().numpy()
            y = torch.cat(predicted_labels).detach().cpu().numpy().ravel()
            n_values = np.max(y) + 1
            y = np.eye(n_values)[y]
            enc_layers_unlab[k] = (X, y)

        besov_index = np.empty((num_layers, len(unlab_inds)))
        for j, ((k_lab, v_lab), (k_unlab, v_unlab)) in enumerate(
            zip(enc_layers.items(), enc_layers_unlab.items())
        ):
            print(f"Layers: ({k_lab}, {k_unlab})")
            X_lab, y_lab = v_lab
            X_unlab, y_unlab = v_unlab

            for i in range(len(unlab_inds)):
                x_i = X_unlab[i]
                y_i = y_unlab[i]
                X = np.concatenate([X_lab, x_i.reshape(1, -1)], axis=0)
                y = np.concatenate([y_lab, y_i.reshape(1, -1)], axis=0)
                rf = WaveletsForestRegressor()
                rf.fit(X, y)
                alpha, n_wav, err = rf.evaluate_smoothness()
                besov_index[j][i] = alpha

        besov_index = besov_index.mean(axis=0)
        print(besov_index.shape)
        print(besov_index)
        return results
