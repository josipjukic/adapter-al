from re import L
from sched import scheduler
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, f1_score
import torch
import time
from scipy.special import softmax

import torch.nn.functional as F
from models import initialize_language_model


from dataloaders import *
from smoothness.random_forest import WaveletsForestRegressor

from util import logits_to_probs, compute_forgetfulness
from podium import Iterator

import math
import wandb

from datetime import datetime


class Experiment:
    def __init__(self, sampler, train_set, test_set, device, args, meta):
        self.sampler = sampler
        self.args = args
        mask = np.full(len(train_set), True)
        if self.args.load_cartography:
            crt_train = pd.read_csv(
                os.path.join(self.args.load_cartography, f"{self.args.model}.csv"),
                index_col=0,
            )

            # TODO: make generic (independent of number of epochs)
            easy_inds = crt_train[crt_train.correctness == 5]
            amb_inds = crt_train[
                (crt_train.correctness == 2) | (crt_train.correctness == 3)
            ]
            hard_inds = crt_train[
                crt_train.correctness
                < 4
                # (crt_train.correctness == 0) | (crt_train.correctness == 1)
            ]

            # Category proportions
            if self.args.prop_easy < 1.0 and self.args.prop_easy >= 0.0:
                easy_remove = easy_inds.sample(frac=1.0 - self.args.prop_easy)
                mask[easy_remove.index] = False

            if self.args.prop_amb < 1.0 and self.args.prop_amb >= 0.0:
                amb_remove = amb_inds.sample(frac=1.0 - self.args.prop_amb)
                mask[amb_remove.index] = False

            if self.args.prop_hard < 1.0 and self.args.prop_hard >= 0.0:
                hard_remove = hard_inds.sample(frac=1.0 - self.args.prop_hard)
                mask[hard_remove.index] = False

        if self.args.load_pvi:
            pvi_train = pd.read_csv(
                os.path.join(self.args.load_pvi, f"pvi_{self.args.model}.csv"),
                index_col=0,
            )

            # Category proportions
            if self.args.pvi_threshold is not None:
                pvi_remove = pvi_train[pvi_train.pvi < self.args.pvi_threshold]
                mask[pvi_remove.index] = False

        indices, *_ = np.where(mask)
        self.train_set = train_set[indices]
        self.test_set = test_set
        self.batch_size = args.batch_size
        self.device = device
        self.meta = meta

        print(self.meta)

        self.test_iter = make_iterable(
            self.test_set,
            self.device,
            batch_size=self.batch_size,
            train=False,
        )

        self.test_lengths = self.extract_test_lengths()
        self.test_id_mapping = self.get_test_id_mapping()

        self.ind2id, self.id2ind = self.create_id_mapping()

    def create_id_mapping(self):
        def cast_to_device(data):
            return torch.tensor(np.array(data), device=self.device)

        iter_ = Iterator(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            matrix_class=cast_to_device,
        )
        ids = []
        for batch in iter_:
            ids.extend([int(id[0]) for id in batch.id])

        ind2id = ids
        id2ind = {ind2id[i]: i for i in range(len(ind2id))}

        return ind2id, id2ind

    def al_loop(
        self,
        create_model_fn,
        criterion,
        warm_start_size,
        query_size,
        tokenizer,
    ):

        selected_examples = []
        selected_inds = None

        # if self.args.load_cartography:
        #     crt_train = pd.read_csv(
        #         os.path.join(self.args.load_cartography, f"{self.args.model}.csv"),
        #         index_col=0,
        #     )
        #     hard_inds = crt_train[
        #         (crt_train.correctness == 0) | (crt_train.correctness == 1)
        #     ]
        #     if self.args.prop_hard < 1.0 and self.args.prop_hard >= 0.0:
        #         hard_remove = hard_inds.sample(frac=1.0 - self.args.prop_hard)
        #         mask = np.full(len(self.train_set), True)
        #         mask[hard_remove.index] = False
        #         self.train_set = self.train_set[mask]

        # Initialize label mask.
        lab_mask = np.full(len(self.train_set), False)
        if self.args.stratified:
            train_iter = make_iterable(
                self.train_set,
                self.device,
                batch_size=self.batch_size,
                train=False,
            )
            labels = []
            with torch.no_grad():
                for batch in train_iter:
                    labels.append(batch.label.squeeze())
            y = torch.cat(labels).cpu().numpy()
            num_classes = y.max() + 1
            n_per_class = warm_start_size // num_classes
            random_inds = []
            for i in range(num_classes):
                y_i = y == i
                indices = np.nonzero(y_i)[0].ravel()
                chosen = np.random.choice(
                    indices, np.minimum(n_per_class, indices.size), replace=False
                )
                random_inds.append(chosen)
            random_inds = np.concatenate(random_inds)
            left = warm_start_size - len(random_inds)
            if left > 0:
                rest = np.setdiff1d(np.arange(len(self.train_set)), random_inds)
                chosen = np.random.choice(rest, left, replace=False)
                random_inds = np.concatenate([random_inds, chosen])

        else:
            random_inds = np.random.choice(
                len(self.train_set), warm_start_size, replace=False
            )
        lab_mask[random_inds] = True

        al_epochs = self.args.al_epochs
        if al_epochs == -1:
            unlab_size = self.args.max_train_size - lab_mask.sum()
            al_epochs = np.int(np.ceil(unlab_size / query_size)) + 1

        results = {
            "train": [],
            "eval": [],
            "untrained": [],
            "labeled": [],
            "selected": [random_inds],
            "grads": [],
        }

        for al_epoch in range(1, al_epochs + 1):
            logging.info(f"AL epoch: {al_epoch}/{al_epochs}")
            results["labeled"].append(lab_mask.sum())

            # 1) Train model with labeled data: fine-tune vs. re-train
            logging.info(
                f"Training on {lab_mask.sum()}/{lab_mask.size} labeled data..."
            )
            # Create new model: re-train scenario.
            if self.args.pretrain:
                model = create_model_fn(self.args, self.meta, pretrained=True)

            else:
                model = create_model_fn(self.args, self.meta)

            model.to(self.device)

            indices, *_ = np.where(lab_mask)
            train_iter = make_iterable(
                self.train_set,
                self.device,
                batch_size=self.batch_size,
                train=True,
                indices=indices,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.args.lr, weight_decay=self.args.l2
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1,
            )
            train_results = []
            eval_results = []
            besov = []
            acc = []
            loss = []

            break_ = False
            for epoch in range(1, self.args.epochs + 1):
                logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
                # a) Train for one epoch
                result_dict_train, logits, y_true, ids = self._train_model(
                    model, optimizer, criterion, train_iter
                )
                if self.args.scheduler:
                    scheduler.step()
                print(result_dict_train)
                train_results.append(result_dict_train)

                # b) Evaluate model (test set)
                eval_result_dict = self._evaluate_model(model)
                # repr_stats = self._representation_stats(model, indices)
                # if selected_inds is not None:
                #     sample_repr = self._representation_stats(model, selected_inds)
                #     eval_result_dict["sample_repr"] = sample_repr

                # eval_result_dict["repr"] = repr_stats
                # besov.append(np.mean(repr_stats["alpha"]))
                acc.append(eval_result_dict["accuracy"])
                loss.append(result_dict_train["loss"])
                eval_results.append(eval_result_dict)

                if break_:
                    break

                # if self.args.besov and epoch > 5:
                #     besov_layers = softmax(repr_stats["alpha"])
                #     besov_middle_layers = besov_layers[4:8]
                #     besov_last_layers = besov_layers[8:]

                #     middle_mass = besov_middle_layers.sum()
                #     end_mass = besov_last_layers.sum()
                #     if middle_mass > end_mass:
                #         break_ = True

            wandb.log(eval_result_dict | {"selected": lab_mask.sum()})

            # if self.args.besov:
            #     num_epochs = max(5, np.argmax(besov) + 1)

            #     model = create_model_fn(self.args, self.meta)
            #     model.to(self.device)
            #     for epoch in range(1, num_epochs + 1):
            #         result_dict_train, logits, y_true, ids = self._train_model(
            #             model, optimizer, criterion, train_iter
            #         )
            #         print(
            #             f"[Besov-optimal epoch]: {epoch}/{num_epochs}",
            #             end="\r",
            #             flush=True,
            #         )
            # elif self.args.best:
            #     num_epochs = np.argmax(acc) + 1

            #     model = create_model_fn(self.args, self.meta)
            #     model.to(self.device)
            #     for epoch in range(1, num_epochs + 1):
            #         result_dict_train, logits, y_true, ids = self._train_model(
            #             model, optimizer, criterion, train_iter
            #         )
            #         print(
            #             f"[Best epoch strategy]: {epoch}/{num_epochs}",
            #             end="\r",
            #             flush=True,
            #         )
            # elif self.args.loss:
            #     num_epochs = np.argmin(loss) + 1

            #     model = create_model_fn(self.args, self.meta)
            #     model.to(self.device)
            #     for epoch in range(1, num_epochs + 1):
            #         result_dict_train, logits, y_true, ids = self._train_model(
            #             model, optimizer, criterion, train_iter
            #         )
            #         print(
            #             f"[Loss strategy]: {epoch}/{num_epochs}",
            #             end="\r",
            #             flush=True,
            #         )

            # 2) Retrieve active sample.
            if not lab_mask.all():
                logging.info("Retrieving AL sample...")
                lab_inds, *_ = np.where(lab_mask)
                unlab_inds, *_ = np.where(~lab_mask)
                if len(unlab_inds) <= query_size:
                    selected_inds = unlab_inds
                else:
                    model.eval()
                    selected_inds = self.sampler.query(
                        query_size=query_size,
                        unlab_inds=unlab_inds,
                        lab_inds=lab_inds,
                        model=model,
                        lab_mask=lab_mask,
                        num_labels=self.meta.num_labels,
                        num_targets=self.meta.num_targets,
                        criterion=criterion,
                        device=self.device,
                    )

                lab_mask[selected_inds] = True
                results["selected"].append(selected_inds)
                logging.info(f"{len(selected_inds)} data points selected.")
                # untrained_sample_repr = self._representation_stats(model, selected_inds)
                # results["untrained"].append(untrained_sample_repr)

            # 3) Store results.
            results["train"].append(train_results)
            results["eval"].append(eval_results)

        return results

   

    def _representation_stats(self, model, indices):

        train_iter = make_iterable(
            self.train_set,
            self.device,
            batch_size=32,
            train=True,
            indices=indices,
        )

        labels = []
        grads = []
        enc = []

        name = model.get_classifier_name()
        clf = getattr(model.classifier, name)
        config = model.classifier.config
        num_layers = config.num_hidden_layers

        enc_layers = {i: [] for i in range(num_layers)}
        results = {}

        for batch in train_iter:

            inputs, _ = batch.text
            labels.append(batch.label)
            inputs.requires_grad = False
            # pad_idx = self.meta.padding_idx
            # attention_mask = inputs != pad_idx

            embedded_tokens = clf.embeddings(inputs)
            embedded_tokens = torch.autograd.Variable(
                embedded_tokens, requires_grad=True
            )
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

            encoded = encoded_all[0][:, 0]
            enc.append(encoded.cpu())

            mean = encoded.mean()
            mean.backward()
            enc_grad = embedded_tokens.grad.data
            grads.append(enc_grad.norm(p=2, dim=(1, 2)))

            torch.cuda.empty_cache()

        grad = torch.cat(grads).cpu()
        enc = torch.cat(enc).cpu()

        results["grad"] = grad

        # Besov smoothness
        alphas = []
        n_wavelets = []
        errors = []
        for k, v in enc_layers.items():
            X = torch.cat(v).detach().numpy()
            y = torch.cat(labels).detach().cpu().numpy().ravel()
            n_values = np.max(y) + 1
            y = np.eye(n_values)[y]

            rf = WaveletsForestRegressor()
            rf.fit(X, y)
            alpha, n_wav, err = rf.evaluate_smoothness()
            alphas.append(alpha)
            n_wavelets.append(n_wav)
            errors.append(err)

        results["alpha"] = alphas
        results["n_wavelets"] = n_wavelets
        results["errors"] = errors

        return results

    def _pretrain_lm(self, tokenizer, epochs=100, lr=2e-5, mlm_prob=0.15):

        lm = initialize_language_model(self.args, self.meta)
        lm.to(self.device)

        optimizer = torch.optim.AdamW(lm.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_iter = make_iterable(
            self.train_set,
            self.device,
            batch_size=self.batch_size,
            train=True,
        )

        # special_tokens_mask = labels == tokenizer.mask_token_id

        for epoch in range(1, epochs + 1):
            logging.info(f"Training epoch: {epoch}/{epochs}")

            total_loss = 0
            for batch_num, batch in enumerate(train_iter, 1):
                inputs, _ = batch.text
                labels = inputs.clone()
                probability_matrix = torch.full(labels.shape, mlm_prob).to(self.device)
                # TODO: account for special tokens (masked them out in the prob matrix)
                # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                masked_indices = (
                    torch.bernoulli(probability_matrix).bool().to(self.device)
                )
                labels[~masked_indices] = -100  # We only compute loss on masked tokens

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                indices_replaced = (
                    torch.bernoulli(torch.full(labels.shape, 0.8))
                    .bool()
                    .to(self.device)
                    & masked_indices
                )
                inputs[indices_replaced] = tokenizer.mask_token_id

                # 10% of the time, we replace masked input tokens with random word
                indices_random = (
                    torch.bernoulli(torch.full(labels.shape, 0.5))
                    .bool()
                    .to(self.device)
                    & masked_indices
                    & ~indices_replaced
                )
                random_words = torch.randint(
                    len(tokenizer), labels.shape, dtype=torch.long
                ).to(self.device)
                inputs[indices_random] = random_words[indices_random]

                optimizer.zero_grad()

                out = lm(inputs).logits
                out = out.view(-1, len(tokenizer))
                loss = criterion(out, labels.view(-1, 1).squeeze())
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lm.parameters(), self.args.clip)
                optimizer.step()

            logging.info(f"Epoch {epoch}: loss = {total_loss}")

        return lm

    
    def unbiased_loop(
        self, create_model_fn, criterion, warm_start_size, query_size, mode, tokenizer
    ):
        # Initialize label mask.
        lab_mask = np.full(len(self.train_set), False)
        # TODO: stratified warm start
        random_inds = np.random.choice(
            len(self.train_set), warm_start_size, replace=False
        )
        lab_mask[random_inds] = True

        al_epochs = self.args.al_epochs
        if al_epochs == -1:
            unlab_size = self.args.max_train_size - lab_mask.sum()
            al_epochs = np.int(np.ceil(unlab_size / query_size)) + 1

        results = {
            "train": [],
            "eval": [],
            "labeled": [],
            "cartography": {"train": [], "test": []},
        }

        N = len(self.train_set)
        weights = torch.tensor(1.0, device=self.device).repeat(N)

        for al_epoch in range(1, al_epochs + 1):
            logging.info(f"AL epoch: {al_epoch}/{al_epochs}")
            results["labeled"].append(lab_mask.sum())

            # 1) Train model with labeled data: fine-tune vs. re-train
            logging.info(
                f"Training on {lab_mask.sum()}/{lab_mask.size} labeled data..."
            )
            # Create new model: re-train scenario.
            model = create_model_fn(self.args, self.meta)
            model.to(self.device)

            indices, *_ = np.where(lab_mask)
            train_iter = make_iterable(
                self.train_set,
                self.device,
                batch_size=self.batch_size,
                train=False,
                indices=indices,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.args.lr, weight_decay=self.args.l2
            )
            # optimizer = torch.optim.Adam(
            #     model.parameters(), self.args.lr, weight_decay=self.args.l2
            # )
            train_results = []
            eval_results = []
            cartography_trends = {
                "train": {"is_correct": [], "true_probs": []},
                "test": {"is_correct": [], "true_probs": []},
            }
            for epoch in range(1, self.args.epochs + 1):
                logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
                # a) Train for one epoch
                train_fn = self._pure if mode == "PURE" else self._lure
                result_dict_train, logits, y_true = train_fn(
                    model,
                    optimizer,
                    criterion,
                    train_iter,
                    indices,
                    weights,
                )
                print(result_dict_train)
                train_results.append(result_dict_train)

                # b) Evaluate model (test set)
                eval_result_dict = self._evaluate_model(model)
                eval_results.append(eval_result_dict)

                # c) Calculate epoch cartography
                logging.info("Calculating cartography...")
                #   i) train set
                is_correct, true_probs = self._cartography_epoch_train(logits, y_true)
                cartography_trends["train"]["is_correct"].append(is_correct)
                cartography_trends["train"]["true_probs"].append(true_probs)

                #   ii) test set
                is_correct, true_probs = self._cartography_epoch_test(model)
                cartography_trends["test"]["is_correct"].append(is_correct)
                cartography_trends["test"]["true_probs"].append(true_probs)

            # 2) Dataset cartography
            logging.info("Computing dataset cartography...")
            cartography_results = {}
            cartography_results["train"] = self._compute_cartography(
                cartography_trends["train"]
            )
            cartography_results["test"] = self._compute_cartography(
                cartography_trends["test"]
            )

            # 2) Retrieve active sample.
            if not lab_mask.all():
                logging.info("Retrieving AL sample...")
                lab_inds, *_ = np.where(lab_mask)
                unlab_inds, *_ = np.where(~lab_mask)
                if len(unlab_inds) <= query_size:
                    selected_inds = unlab_inds
                else:
                    model.eval()
                    selected_inds = self.sampler.query(
                        query_size=query_size,
                        unlab_inds=unlab_inds,
                        lab_inds=lab_inds,
                        model=model,
                        lab_mask=lab_mask,
                        num_labels=self.meta.num_labels,
                        num_targets=self.meta.num_targets,
                        criterion=criterion,
                        device=self.device,
                    )
                    weights = self.sampler.weights(
                        query_size=query_size,
                        unlab_inds=np.arange(N),
                        lab_inds=lab_inds,
                        model=model,
                        lab_mask=lab_mask,
                        num_labels=self.meta.num_labels,
                        num_targets=self.meta.num_targets,
                        criterion=criterion,
                    )
                    weights = torch.tensor(weights, device=self.device)

                lab_mask[selected_inds] = True
                logging.info(f"{len(selected_inds)} data points selected.")

            # 3) Store results.
            results["train"].append(train_results)
            results["eval"].append(eval_results)
            results["cartography"]["train"].append(cartography_results["train"])
            results["cartography"]["test"].append(cartography_results["test"])

        return results

   
    def _pure(self, model, optimizer, criterion, train_iter, lab_inds, weights):
        model.train()

        # criterion => reduction should be set to "none"
        total_loss = 0.0
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        m = 0
        N = len(self.train_set)
        M = lab_inds.size
        for batch_num, batch in enumerate(train_iter, 1):
            t = time.time()

            # Unpack batch & cast to device
            (x, lengths), y = batch.text, batch.label
            y_true_list.append(y.squeeze(0) if y.numel() == 1 else y.squeeze())

            logits, _ = model(x, lengths)
            logit_list.append(logits)

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = Experiment.update_stats(
                accuracy, confusion_matrix, logits, y
            )
            if logits.shape[-1] == 1:
                # binary cross entropy, cast labels to float
                y = y.type(torch.float)

            r_pure = torch.tensor(0.0, device=self.device)
            loss = criterion(
                logits.view(-1, self.meta.num_targets).squeeze(), y.squeeze()
            )

            for loss_i in loss:
                weights_i = weights[m:]
                # perhaps replace with log softmax
                q = F.softmax(weights_i, dim=0)[0]
                r_pure += loss_i * (1 / (N * q) + (M - m) / N)
                m += 1

            total_loss += float(r_pure)

            optimizer.zero_grad()
            r_pure.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()

            print(
                "[Batch]: {}/{} in {:.5f} seconds".format(
                    batch_num, len(train_iter), time.time() - t
                ),
                end="\r",
                flush=True,
            )

        loss = total_loss / len(train_iter)
        result_dict = {"loss": loss}
        logit_tensor = torch.cat(logit_list)
        y_true = torch.cat(y_true_list)
        return result_dict, logit_tensor, y_true

    def _lure(self, model, optimizer, criterion, train_iter, lab_inds, weights):
        model.train()

        # criterion => reduction should be set to "none"
        total_loss = 0.0
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        m = 0
        N = len(self.train_set)
        M = lab_inds.size
        for batch_num, batch in enumerate(train_iter, 1):
            t = time.time()

            # Unpack batch & cast to device
            (x, lengths), y = batch.text, batch.label
            y_true_list.append(y.squeeze(0) if y.numel() == 1 else y.squeeze())

            logits, _ = model(x, lengths)
            logit_list.append(logits)

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = Experiment.update_stats(
                accuracy, confusion_matrix, logits, y
            )
            if logits.shape[-1] == 1:
                # binary cross entropy, cast labels to float
                y = y.type(torch.float)

            r_lure = torch.tensor(0.0, device=self.device)
            loss = criterion(
                logits.view(-1, self.meta.num_targets).squeeze(), y.squeeze()
            )

            for loss_i in loss:
                weights_i = weights[m:]
                # perhaps replace with log softmax
                q = F.softmax(weights_i, dim=0)[0]
                v_i = 1 + (N - M) / (N - m) * (1 / (q * (N - m + 1)) - 1)
                r_lure += loss_i * v_i
                m += 1

            total_loss += float(r_lure)

            optimizer.zero_grad()
            r_lure.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()

            print(
                "[Batch]: {}/{} in {:.5f} seconds".format(
                    batch_num, len(train_iter), time.time() - t
                ),
                end="\r",
                flush=True,
            )

        loss = total_loss / len(train_iter)
        result_dict = {"loss": loss}
        logit_tensor = torch.cat(logit_list)
        y_true = torch.cat(y_true_list)
        return result_dict, logit_tensor, y_true


    def _train_model(self, model, optimizer, criterion, train_iter):
        model.train()

        total_loss = 0.0
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        ids = []
        for batch_num, batch in enumerate(train_iter, 1):
            t = time.time()

            optimizer.zero_grad()

            ids.extend([int(id[0]) for id in batch.id])

            # Unpack batch & cast to device
            if self.meta.pair_sequence:
                (x_premise, premise_lengths) = batch.premise
                (x_hypothesis, hypothesis_lengths) = batch.hypothesis
            else:
                (x, lengths) = batch.text

            y = batch.label
            y_true_list.append(y.squeeze(0) if y.numel() == 1 else y.squeeze())

            if self.meta.pair_sequence:
                # PSQ
                lengths = (premise_lengths, hypothesis_lengths)
                logits, return_dict = model(x_premise, x_hypothesis, lengths)
            else:
                # SSQ
                logits, return_dict = model(x, lengths)
            logit_list.append(logits)

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = Experiment.update_stats(
                accuracy, confusion_matrix, logits, y
            )
            if logits.shape[-1] == 1:
                # binary cross entropy, cast labels to float
                y = y.type(torch.float)

            loss = criterion(
                logits.view(-1, self.meta.num_targets).squeeze(), y.squeeze()
            )

            total_loss += float(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()

            print(
                "[Batch]: {}/{} in {:.5f} seconds".format(
                    batch_num, len(train_iter), time.time() - t
                ),
                end="\r",
                flush=True,
            )

        loss = total_loss / len(train_iter)
        result_dict = {"loss": loss}
        logit_tensor = torch.cat(logit_list)
        y_true = torch.cat(y_true_list)
        return result_dict, logit_tensor, y_true, ids

    def _online_coding(self, model, optimizer, criterion, train_iter, halt):
        model.train()

        total_loss = 0.0
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        num_examples = 0
        halt = min(halt, len(train_iter))
        block_loss = []
        for batch_num, batch in enumerate(train_iter, 1):
            t = time.time()

            optimizer.zero_grad()

            # Unpack batch & cast to device
            (x, lengths), y = batch.text, batch.label

            # y needs to be a 1D tensor for xent(batch_size)
            y = y.squeeze()
            y_true_list.append(y)

            num_examples += y.shape[0]

            logits, return_dict = model(x, lengths)

            logit_list.append(logits)

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = Experiment.update_stats(
                accuracy, confusion_matrix, logits, y
            )
            if logits.shape[-1] == 1:
                # binary cross entropy, cast labels to float
                y = y.type(torch.float)

            losses = criterion(logits.view(-1, self.meta.num_targets).squeeze(), y)
            block_loss.append(losses.detach().cpu())

            loss = losses.mean()

            total_loss += float(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()

            print(
                "[Batch]: {}/{} in {:.5f} seconds".format(
                    batch_num, halt, time.time() - t
                ),
                end="\r",
                flush=True,
            )

            if batch_num >= halt:
                break

        loss = total_loss / len(train_iter)

        logging.info(f"[Train acc]: {accuracy/num_examples*100:.3f}%")
        block_loss_tensor = torch.cat(block_loss)

        return block_loss_tensor.mean()

    def _evaluate_model(self, model):
        model.eval()

        data = self.test_iter
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        with torch.inference_mode():
            for batch_num, batch in enumerate(data):
                # if batch_num > 100: break # checking beer imitation

                t = time.time()

                # Unpack batch & cast to device
                (x, lengths), y = batch.text, batch.label

                y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)

                y_true_list.append(y.cpu())

                logits, _ = model(x, lengths)
                logit_list.append(logits.cpu())

                # Bookkeeping and cast label to float
                accuracy, confusion_matrix = Experiment.update_stats(
                    accuracy, confusion_matrix, logits, y
                )

                print(
                    "[Batch]: {}/{} in {:.5f} seconds".format(
                        batch_num, len(data), time.time() - t
                    ),
                    end="\r",
                    flush=True,
                )

        logit_tensor = torch.cat(logit_list)
        y_true = torch.cat(y_true_list)
        probs = logits_to_probs(logit_tensor)
        true_probs = (
            probs.gather(dim=1, index=y_true.unsqueeze(dim=1)).squeeze().numpy()
        )
        y_pred = torch.argmax(probs, dim=1).numpy()
        f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")

        logging.info(
            "[Accuracy]: {}/{} : {:.3f}%".format(
                accuracy,
                len(self.test_set),
                accuracy / len(self.test_set) * 100,
            )
        )
        logging.info(f"[F1-micro]: {f1_micro:.3f}")
        logging.info(f"[F1-macro]: {f1_macro:.3f}")
        logging.info(confusion_matrix)

        result_dict = {
            "accuracy": accuracy / len(self.test_set),
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
        }

        return result_dict

    def _evaluate_entropy(self, model, y_dist):
        model.eval()

        N = len(self.train_set)
        iter_ = make_iterable(
            self.train_set,
            self.device,
            batch_size=1,
            train=False,  # [Debug] was False
        )

        hy = 0
        hyx = 0
        pvis = []
        with torch.inference_mode():
            ids = []
            for batch_num, batch in enumerate(iter_):
                t = time.time()

                ids.extend([int(id[0]) for id in batch.id])

                # Unpack batch & cast to device
                (x, lengths), y = batch.text, batch.label

                y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)
                probs = model.predict_probs(x, lengths).ravel()
                prob_i = probs[y]
                y_prime_i = y_dist[y]

                hy -= 1 / N * torch.log(y_prime_i)
                hyx -= 1 / N * torch.log(prob_i)
                pvi = -torch.log(y_prime_i) + torch.log(prob_i)
                pvis.append(pvi)

                print(
                    "[Batch]: {}/{} in {:.5f} seconds".format(
                        batch_num, len(iter_), time.time() - t
                    ),
                    end="\r",
                    flush=True,
                )

        pvi_tensor = torch.tensor(pvis)

        # Preserve instance ordering
        inds = np.argsort([self.id2ind[id] for id in ids])
        pvi_tensor = pvi_tensor[inds]

        return pvi_tensor

    @staticmethod
    def update_stats(accuracy, confusion_matrix, logits, y):
        if logits.shape[-1] == 1:
            # BCE, need to check ge 0 (or gt 0?)
            max_ind = torch.ge(logits, 0).type(torch.long).squeeze()
        else:
            _, max_ind = torch.max(logits, 1)

        equal = torch.eq(max_ind, y)
        correct = int(torch.sum(equal))
        if len(max_ind.shape) == 0:
            # only one element here? is this even possible?
            confusion_matrix[y, max_ind] += 1
        else:
            for j, i in zip(max_ind, y):
                confusion_matrix[int(i), int(j)] += 1

        return accuracy + correct, confusion_matrix

    def _cartography_epoch_train(self, logits, y_true, ids):
        logits = logits.cpu()
        y_true = y_true.cpu()
        probs = logits_to_probs(logits)
        true_probs = probs.gather(dim=1, index=y_true.unsqueeze(dim=1)).squeeze()
        y_pred = torch.argmax(probs, dim=1)
        is_correct = y_pred == y_true

        # Preserve instance ordering
        inds = np.argsort([self.id2ind[id] for id in ids])

        return is_correct[inds], true_probs[inds]

    def _cartography_epoch_test(self, model, ids):
        model.train()

        data = self.test_iter

        logit_list = []
        y_true_list = []
        with torch.inference_mode():
            for batch_num, batch in enumerate(data):
                # Unpack batch & cast to device
                (x, lengths), y = batch.text, batch.label

                y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)
                y_true_list.append(y.cpu())

                logits, _ = model(x, lengths)
                logit_list.append(logits.cpu())

        logit_tensor = torch.cat(logit_list)
        y_true = torch.cat(y_true_list)
        probs = logits_to_probs(logit_tensor)
        true_probs = probs.gather(dim=1, index=y_true.unsqueeze(dim=1)).squeeze()
        y_pred = torch.argmax(probs, dim=1)
        is_correct = y_pred == y_true

        inds = np.argsort([self.id2ind[id] for id in ids])

        return is_correct[inds], true_probs[inds]

    def _compute_cartography(self, trends):
        cartography_results = {}

        is_correct = torch.stack(trends["is_correct"])
        true_probs = torch.stack(trends["true_probs"])

        cartography_results["correctness"] = (
            is_correct.sum(dim=0).squeeze().detach().numpy()
        )
        cartography_results["confidence"] = (
            true_probs.mean(dim=0).squeeze().detach().numpy()
        )
        cartography_results["variability"] = (
            true_probs.std(dim=0).squeeze().detach().numpy()
        )
        cartography_results["forgetfulness"] = compute_forgetfulness(is_correct).numpy()
        conf = cartography_results["confidence"]
        cartography_results["threshold_closeness"] = conf * (1 - conf)

        return cartography_results

    def cartography(
        self,
        create_model_fn,
        criterion,
    ):
        lab_mask = np.full(len(self.train_set), True)

        # 1) Train model on labeled data.
        logging.info(f"Training on {lab_mask.sum()}/{lab_mask.size} labeled data...")
        # Create new model: re-train scenario.
        model = create_model_fn(self.args, self.meta)
        model.to(self.device)

        indices, *_ = np.where(lab_mask)
        train_iter = make_iterable(
            self.train_set,
            self.device,
            batch_size=self.batch_size,
            train=True,  # [Debug] was False
            indices=indices,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            self.args.lr,
            weight_decay=self.args.l2,
        )

        cartography_trends = {
            "train": {"is_correct": [], "true_probs": []},
            # "test": {"is_correct": [], "true_probs": []},
        }
        for epoch in range(1, self.args.epochs + 1):
            logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
            # a) Train for one epoch
            result_dict_train, logits, y_true, ids = self._train_model(
                model, optimizer, criterion, train_iter
            )

            # b) Calculate epoch cartography
            logging.info("Calculating cartography...")
            #   i) train set
            is_correct, true_probs = self._cartography_epoch_train(logits, y_true, ids)
            cartography_trends["train"]["is_correct"].append(is_correct)
            cartography_trends["train"]["true_probs"].append(true_probs)

            # #   ii) test set
            # is_correct, true_probs = self._cartography_epoch_test(model, ids)
            # cartography_trends["test"]["is_correct"].append(is_correct)
            # cartography_trends["test"]["true_probs"].append(true_probs)

        # 2) Dataset cartography
        logging.info("Computing dataset cartography...")
        cartography_results = {}
        cartography_results["train"] = self._compute_cartography(
            cartography_trends["train"]
        )
        # cartography_results["test"] = self._compute_cartography(
        #     cartography_trends["test"]
        # )

        # 3) MC-Evaluate local smoothness
        smoothness_results = self._evaluate_smoothness(model)

        # 3) Store results.
        results = {}
        results["cartography"] = {}
        results["cartography"]["train"] = cartography_results["train"]
        # results["cartography"]["test"] = cartography_results["test"]
        results["smoothness"] = {}
        results["smoothness"]["logit_variance"] = smoothness_results["logit_variance"]

        return results

    def calculate_predictive_entropy(
        self,
        create_model_fn,
        criterion,
    ):

        # 1) Train model on labeled data.
        model = create_model_fn(self.args, self.meta)
        model.to(self.device)

        train_iter = make_iterable(
            self.train_set,
            self.device,
            batch_size=self.batch_size,
            train=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            self.args.lr,
            weight_decay=self.args.l2,
        )

        for epoch in range(1, self.args.epochs + 1):
            logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
            # a) Train for one epoch
            _, _, y_true, _ = self._train_model(model, optimizer, criterion, train_iter)
        y_dist = y_true.bincount() / y_true.shape[0]

        # 2) Calculate predictive entropy
        pvi = self._evaluate_entropy(model, y_dist)

        return pvi

    def calculate_mdl(
        self,
        create_model_fn,
        criterion,
    ):
        model = create_model_fn(self.args, self.meta)
        model.to(self.device)

        def cast_to_device(data):
            return torch.tensor(np.array(data), device=self.device)

        train_iter = Iterator(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            matrix_class=cast_to_device,
        )

        train_iter = list(train_iter)

        ids = []
        for batch in train_iter:
            ids.extend([int(id[0]) for id in batch.id])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            self.args.lr,
            weight_decay=self.args.l2,
        )

        N = len(self.train_set)
        num_blocks = torch.zeros(N)
        max_power = math.ceil(math.log2(N))
        halt_list = [2 ** x // self.batch_size for x in range(6, max_power + 1)]
        codelenghts = []

        block_losses = []
        for halt in halt_list:
            for epoch in range(1, self.args.epochs + 1):
                logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
                # Train for one epoch
                block_loss = self._online_coding(
                    model, optimizer, criterion, train_iter, halt
                )
                if epoch == self.args.epochs:
                    block_losses.append(block_loss)
                    codelenghts.append(block_loss / np.log(2))

        return codelenghts, halt_list

    def extract_test_lengths(self):
        len_list = []
        for batch in self.test_iter:
            _, lengths = batch.text
            len_list.append(lengths.cpu())

        return torch.cat(len_list)

    def get_test_id_mapping(self):
        mapping_list = []
        for batch in self.test_iter:
            mapping_list.extend(batch.id)

        return [int(id) for ids in mapping_list for id in ids]

    def extract_train_lengths(self):
        len_list = []
        train_iter = make_iterable(
            self.train_set,
            self.device,
            batch_size=self.batch_size,
            train=False,
        )
        for batch in train_iter:
            _, lengths = batch.text
            len_list.append(lengths.cpu())

        return torch.cat(len_list)

    def get_train_id_mapping(self):
        mapping_list = []
        train_iter = make_iterable(
            self.train_set,
            self.device,
            batch_size=self.batch_size,
            train=False,
        )
        for batch in train_iter:
            mapping_list.extend(batch.id)

        return [int(id) for ids in mapping_list for id in ids]
