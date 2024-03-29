import numpy as np
import logging
from sklearn.metrics import f1_score
import torch
import time

import torch.nn.functional as F
from models import initialize_language_model


from dataloaders import *
from smoothness.random_forest import WaveletsForestRegressor

from util import logits_to_probs, compute_forgetfulness
from text import SingleSequenceIterator
from dataloaders import make_iterable

import math
import wandb


class Experiment:
    def __init__(self, sampler, train_set, test_set, device, args, meta):
        self.sampler = sampler
        self.args = args
        mask = np.full(len(train_set), True)

        indices, *_ = np.where(mask)
        self.train_set = train_set[indices]
        self.test_set = test_set
        self.batch_size = args.batch_size
        self.device = device
        self.meta = meta

        self.test_iter = make_iterable(
            self.test_set,
            self.device,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # self.test_lengths = self.extract_test_lengths()
        # self.test_id_mapping = self.get_test_id_mapping()
        # self.ind2id, self.id2ind = self.create_id_mapping()

    # def create_id_mapping(self):
    #     def cast_to_device(data):
    #         return torch.tensor(np.array(data), device=self.device)

    #     iter_ = Iterator(
    #         self.train_set,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         matrix_class=cast_to_device,
    #     )
    #     ids = []
    #     for batch in iter_:
    #         ids.extend([int(id[0]) for id in batch.id])

    #     ind2id = ids
    #     id2ind = {ind2id[i]: i for i in range(len(ind2id))}

    #     return ind2id, id2ind

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

        # Initialize label mask.
        lab_mask = np.full(len(self.train_set), False)
        if self.args.stratified:
            train_iter = make_iterable(
                self.train_set,
                self.device,
                batch_size=self.batch_size,
                shuffle=False,
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
            model = create_model_fn(self.args, self.meta, self.args.pretrain)

            model.to(self.device)

            indices, *_ = np.where(lab_mask)
            train_iter = make_iterable(
                self.train_set,
                self.device,
                batch_size=self.batch_size,
                shuffle=True,
                indices=indices,
                bucket=True,
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
            acc = []
            loss = []

            for epoch in range(1, self.args.epochs + 1):
                logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
                # a) Train for one epoch
                result_dict_train, logits, y_true, ids = self._train_model(
                    model, optimizer, criterion, train_iter, tokenizer
                )
                if self.args.scheduler:
                    scheduler.step()
                print(result_dict_train)
                train_results.append(result_dict_train)

                # b) Evaluate model (test set)
                eval_result_dict = self._evaluate_model(model, tokenizer)
                acc.append(eval_result_dict["accuracy"])
                loss.append(result_dict_train["loss"])
                eval_results.append(eval_result_dict)

            wandb.log(eval_result_dict | {"selected": lab_mask.sum()})

            # 2) Retrieve active sample.
            if not lab_mask.all() and al_epoch < self.args.epochs:
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

            # 3) Store results.
            results["train"].append(train_results)
            results["eval"].append(eval_results)

        return results

    def forget_profile(
        self,
        create_model_fn,
        criterion,
        warm_start_size,
        query_size,
        tokenizer,
        df_diagnostics,
    ):

        # Initialize label mask.
        lab_mask = np.full(len(self.train_set), False)
        if self.args.stratified:
            train_iter = make_iterable(
                self.train_set,
                self.device,
                batch_size=self.batch_size,
                shuffle=False,
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
            "trained_on": [random_inds],
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
            model = create_model_fn(self.args, self.meta, self.args.pretrain)

            model.to(self.device)

            indices, *_ = np.where(lab_mask)
            df_sample = df_diagnostics.iloc[indices]
            df_sample = df_sample[df_sample.forgetfulness < 3]
            new_indices = df_sample.index.tolist()
            results["trained_on"].append(new_indices)
            train_iter = make_iterable(
                self.train_set,
                self.device,
                batch_size=self.batch_size,
                shuffle=True,
                indices=new_indices,
                bucket=True,
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
            acc = []
            loss = []

            for epoch in range(1, self.args.epochs + 1):
                logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
                # a) Train for one epoch
                result_dict_train, logits, y_true, ids = self._train_model(
                    model, optimizer, criterion, train_iter, tokenizer
                )
                if self.args.scheduler:
                    scheduler.step()
                print(result_dict_train)
                train_results.append(result_dict_train)

                # b) Evaluate model (test set)
                eval_result_dict = self._evaluate_model(model, tokenizer)
                acc.append(eval_result_dict["accuracy"])
                loss.append(result_dict_train["loss"])
                eval_results.append(eval_result_dict)

            wandb.log(eval_result_dict | {"selected": lab_mask.sum()})

            # 2) Retrieve active sample.
            if not lab_mask.all() and al_epoch < self.args.epochs:
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

            # 3) Store results.
            results["train"].append(train_results)
            results["eval"].append(eval_results)

        return results

    # def _representation_stats(self, model, indices):

    #     train_iter = make_iterable(
    #         self.train_set,
    #         self.device,
    #         batch_size=32,
    #         shuffle=True,
    #         indices=indices,
    #     )

    #     labels = []
    #     grads = []
    #     enc = []

    #     name = model.get_classifier_name()
    #     clf = getattr(model.classifier, name)
    #     config = model.classifier.config
    #     num_layers = config.num_hidden_layers

    #     enc_layers = {i: [] for i in range(num_layers)}
    #     results = {}

    #     for batch in train_iter:

    #         inputs, _ = batch.text
    #         labels.append(batch.label)
    #         inputs.requires_grad = False
    #         # pad_idx = self.meta.padding_idx
    #         # attention_mask = inputs != pad_idx

    #         embedded_tokens = clf.embeddings(inputs)
    #         embedded_tokens = torch.autograd.Variable(
    #             embedded_tokens, requires_grad=True
    #         )
    #         # head_mask = attention_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
    #         # head_mask = head_mask.expand(
    #         #     config.num_hidden_layers,
    #         #     -1,
    #         #     config.num_attention_heads,
    #         #     -1,
    #         #     attention_mask.shape[1],
    #         # )
    #         encoded_all = clf.encoder(
    #             embedded_tokens,
    #             output_hidden_states=True,
    #             # head_mask=head_mask,
    #             # attention_mask=attention_mask,
    #         )
    #         # Skip the embedding layer [1:]
    #         for i, enc_layer in enumerate(encoded_all[1][1:]):
    #             enc_layers[i].append(enc_layer[:, 0].cpu())

    #         encoded = encoded_all[0][:, 0]
    #         enc.append(encoded.cpu())

    #         mean = encoded.mean()
    #         mean.backward()
    #         enc_grad = embedded_tokens.grad.data
    #         grads.append(enc_grad.norm(p=2, dim=(1, 2)))

    #         torch.cuda.empty_cache()

    #     grad = torch.cat(grads).cpu()
    #     enc = torch.cat(enc).cpu()

    #     results["grad"] = grad

    #     # Besov smoothness
    #     alphas = []
    #     n_wavelets = []
    #     errors = []
    #     for k, v in enc_layers.items():
    #         X = torch.cat(v).detach().numpy()
    #         y = torch.cat(labels).detach().cpu().numpy().ravel()
    #         n_values = np.max(y) + 1
    #         y = np.eye(n_values)[y]

    #         rf = WaveletsForestRegressor()
    #         rf.fit(X, y)
    #         alpha, n_wav, err = rf.evaluate_smoothness()
    #         alphas.append(alpha)
    #         n_wavelets.append(n_wav)
    #         errors.append(err)

    #     results["alpha"] = alphas
    #     results["n_wavelets"] = n_wavelets
    #     results["errors"] = errors

    #     return results

    def _pretrain_lm(self, tokenizer, epochs=100, lr=2e-5, mlm_prob=0.15):

        lm = initialize_language_model(self.args, self.meta)
        lm.to(self.device)

        optimizer = torch.optim.AdamW(lm.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_iter = make_iterable(
            self.train_set,
            self.device,
            batch_size=self.batch_size,
            shuffle=True,
        )

        iter_ = SingleSequenceIterator(
            train_iter, tokenizer=tokenizer, device=self.device
        )

        # special_tokens_mask = labels == tokenizer.mask_token_id

        for epoch in range(1, epochs + 1):
            logging.info(f"Training epoch: {epoch}/{epochs}")

            total_loss = 0
            for batch_num, batch in enumerate(iter_, 1):
                # Unpack batch & cast to device
                if self.meta.pair_sequence:
                    (x_sequence1, _) = batch.sequence1
                    (x_sequence2, _) = batch.sequence2
                    sep = tokenizer.sep_token_id
                    n = x_sequence1.shape[0]
                    sep_tensor = (
                        torch.tensor(sep, device=self.device).repeat(n).reshape(n, 1)
                    )
                    inputs = torch.cat([x_sequence1, sep_tensor, x_sequence2], dim=1)
                else:
                    inputs = batch.input_ids
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

    def _train_model(self, model, optimizer, criterion, train_iter, tokenizer):
        model.train()

        total_loss = 0.0
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        ids = []

        iter_ = SingleSequenceIterator(
            train_iter, tokenizer=tokenizer, device=self.device
        )
        for batch_num, batch in enumerate(iter_, 1):
            t = time.time()

            optimizer.zero_grad()

            ids.extend([id for id in batch.instance_ids])

            # Unpack batch & cast to device
            if self.meta.pair_sequence:
                (x_sequence1, sequence1_lengths) = batch.sequence1
                (x_sequence2, sequence2_lengths) = batch.sequence2
                sep = tokenizer.sep_token_id
                n = x_sequence1.shape[0]
                sep_tensor = (
                    torch.tensor(sep, device=self.device).repeat(n).reshape(n, 1)
                )
                x = torch.cat([x_sequence1, sep_tensor, x_sequence2], dim=1)
                lengths = sequence1_lengths + sequence2_lengths + 1
            else:
                x = batch.input_ids

            y = batch.target
            y_true_list.append(y.squeeze(0) if y.numel() == 1 else y.squeeze())
            output, return_dict = model(
                x,
                attention_mask=batch.attention_mask,
                token_type_ids=batch.token_type_ids,
            )
            logits = output.logits
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

    def _evaluate_model(self, model, tokenizer):
        model.eval()

        data = self.test_iter
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        iter_ = SingleSequenceIterator(data, tokenizer=tokenizer, device=self.device)
        with torch.inference_mode():
            for batch_num, batch in enumerate(iter_):
                # if batch_num > 100: break # checking beer imitation

                t = time.time()

                # Unpack batch & cast to device
                if self.meta.pair_sequence:
                    (x_sequence1, sequence1_lengths) = batch.sequence1
                    (x_sequence2, sequence2_lengths) = batch.sequence2
                    sep = tokenizer.sep_token_id
                    n = x_sequence1.shape[0]
                    sep_tensor = (
                        torch.tensor(sep, device=self.device).repeat(n).reshape(n, 1)
                    )
                    x = torch.cat([x_sequence1, sep_tensor, x_sequence2], dim=1)
                else:
                    x = batch.input_ids

                y = batch.target
                y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)

                y_true_list.append(y.cpu())

                output, _ = model(
                    x,
                    attention_mask=batch.attention_mask,
                    token_type_ids=batch.token_type_ids,
                )
                loss, logits = output.loss, output.logits

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

    def calculate_predictive_entropy(
        self,
        create_model_fn,
        criterion,
        tokenizer,
    ):

        # 1) Train model on labeled data.
        model = create_model_fn(self.args, self.meta)
        model.to(self.device)

        train_iter = make_iterable(
            self.train_set,
            self.device,
            batch_size=self.batch_size,
            shuffle=False,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            self.args.lr,
            weight_decay=self.args.l2,
        )

        for epoch in range(1, self.args.epochs + 1):
            logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
            # a) Train for one epoch
            _, _, y_true, _ = self._train_model(
                model, optimizer, criterion, train_iter, tokenizer
            )
        y_dist = y_true.bincount() / y_true.shape[0]

        # 2) Calculate predictive entropy
        pvi = self._evaluate_entropy(model, y_dist, tokenizer)

        return pvi

    def _evaluate_entropy(self, model, y_dist, tokenizer):
        model.eval()

        N = len(self.train_set)
        iter = make_iterable(
            self.train_set,
            self.device,
            batch_size=1,
            shuffle=False,
        )
        iter_ = SingleSequenceIterator(iter, tokenizer=tokenizer, device=self.device)
        pvis = []
        with torch.inference_mode():
            ids = []
            for batch_num, batch in enumerate(iter_):
                t = time.time()

                # Unpack batch & cast to device
                x, y = batch.input_ids, batch.target

                y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)
                probs = model.predict_probs(
                    x,
                    attention_mask=batch.attention_mask,
                    token_type_ids=batch.token_type_ids,
                ).ravel()
                prob_i = probs[y]
                y_prime_i = y_dist[y]

                hy = -torch.log2(y_prime_i)
                hyx = -torch.log2(prob_i)
                pvi = hy - hyx
                pvis.append(pvi)

                # if batch_num == 0:
                #     print("y prime", y_prime_i)
                #     print("prob", prob_i)
                #     print("Hy:", hy)
                #     print("Hyx:", hyx)
                #     print("PVI", pvi)

        pvi_tensor = torch.tensor(pvis)

        return pvi_tensor

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

    def _cartography_epoch_train(self, logits, y_true, ids):
        logits = logits.cpu()
        y_true = y_true.cpu()
        probs = logits_to_probs(logits)
        true_probs = probs.gather(dim=1, index=y_true.unsqueeze(dim=1)).squeeze()
        y_pred = torch.argmax(probs, dim=1)
        is_correct = y_pred == y_true

        return is_correct, true_probs

    def cartography(
        self,
        create_model_fn,
        criterion,
        tokenizer,
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
            shuffle=False,
            indices=indices,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            self.args.lr,
            weight_decay=self.args.l2,
        )

        cartography_trends = {
            "train": {"is_correct": [], "true_probs": []},
        }
        for epoch in range(1, self.args.epochs + 1):
            logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
            # a) Train for one epoch
            result_dict_train, logits, y_true, ids = self._train_model(
                model, optimizer, criterion, train_iter, tokenizer
            )

            # b) Calculate epoch cartography
            logging.info("Calculating cartography...")
            #   i) train set
            is_correct, true_probs = self._cartography_epoch_train(logits, y_true, ids)
            cartography_trends["train"]["is_correct"].append(is_correct)
            cartography_trends["train"]["true_probs"].append(true_probs)

        # 2) Dataset cartography
        logging.info("Computing dataset cartography...")
        cartography_dict = self._compute_cartography(cartography_trends["train"])

        return pd.DataFrame(cartography_dict)

    def calculate_mdl(
        self,
        create_model_fn,
        criterion,
    ):

        # 1) Train model on labeled data.
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
        max_power = 12
        halt_list = [2**x // self.batch_size for x in range(6, max_power + 1)]
        codelenghts = []

        block_losses = []
        for halt in halt_list:
            for epoch in range(1, self.args.epochs + 1):
                logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
                # Train for one epoch
                train_loss = self._online_coding(
                    model, optimizer, criterion, train_iter, halt
                )
                block_loss = self._evaluate_online_coding(model, optimizer, criterion)
                if epoch == self.args.epochs:
                    block_losses.append(block_loss)
                    codelenghts.append(block_loss / np.log(2))

        return codelenghts, halt_list

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

    def _evaluate_online_coding(self, model, optimizer, criterion):
        model.eval()

        total_loss = 0.0
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        num_examples = 0
        total_loss = 0
        with torch.inference_mode():
            for batch_num, batch in enumerate(self.test_iter, 1):
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

                loss = criterion(logits.view(-1, self.meta.num_targets).squeeze(), y)
                loss_item = loss.sum().item()
                total_loss += loss_item

        return total_loss / num_examples

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

    # def extract_train_lengths(self):
    #     len_list = []
    #     train_iter = make_iterable(
    #         self.train_set,
    #         self.device,
    #         batch_size=self.batch_size,
    #         train=False,
    #     )
    #     for batch in train_iter:
    #         _, lengths = batch.text
    #         len_list.append(lengths.cpu())

    #     return torch.cat(len_list)

    # def get_train_id_mapping(self):
    #     mapping_list = []
    #     train_iter = make_iterable(
    #         self.train_set,
    #         self.device,
    #         batch_size=self.batch_size,
    #         train=False,
    #     )
    #     for batch in train_iter:
    #         mapping_list.extend(batch.id)

    #     return [int(id) for ids in mapping_list for id in ids]
