from abc import ABC, abstractmethod
import numpy as np
import torch

from dataloaders import make_iterable


class Sampler(ABC):
    def __init__(self, dataset, batch_size, device, meta):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.meta = meta

    @abstractmethod
    def query(self, query_size, *args, **kwargs):
        pass

    def _forward_iter(self, indices, forward_fn):
        iter = make_iterable(
            self.dataset, self.device, batch_size=self.batch_size, indices=indices
        )
        out_list = []
        for batch in iter:
            if self.meta.pair_sequence:
                (x_sequence1, sequence1_lengths) = batch.sequence1
                (x_sequence2, sequence2_lengths) = batch.sequence2
                lengths = (sequence1_lengths, sequence2_lengths)
                out = forward_fn(x_sequence1, x_sequence2, lengths)
            else:
                (x, lengths) = batch.text
                out = forward_fn(x, lengths=lengths)
            out_list.append(out)

        res = torch.cat(out_list)
        return res

    def _predict_probs_dropout(self, model, n_drop, indices, num_labels):
        model.train()

        probs = torch.zeros([len(indices), num_labels]).to(self.device)

        iter = make_iterable(
            self.dataset, self.device, batch_size=self.batch_size, indices=indices
        )

        # Dropout approximation for output probs.
        for _ in range(n_drop):
            index = 0
            for batch in iter:
                x, lengths = batch.text
                probs_i = model.predict_probs(x, lengths=lengths)
                start = index
                end = start + x.shape[0]
                probs[start:end] += probs_i
                index = end

        probs /= n_drop

        return probs

    def _get_grad_embedding(
        self,
        model,
        criterion,
        indices,
        num_targets,
        grad_embedding_type="bias_linear",
    ):

        # cpu_device = torch.device("cpu")
        # model.to(cpu_device)
        if len(indices) > 500:
            indices = np.random.choice(indices, 500, replace=False)
        iter = make_iterable(
            self.dataset, self.device, batch_size=self.batch_size, indices=indices
        )

        encoder_dim = model.get_encoder_dim()

        # Create the tensor to return depending on the grad_embedding_type, which can have bias only,
        # linear only, or bias and linear.
        if grad_embedding_type == "bias":
            grad_embedding = torch.zeros([len(indices), num_targets])
        elif grad_embedding_type == "linear":
            grad_embedding = torch.zeros([len(indices), encoder_dim * num_targets])
        elif grad_embedding_type == "bias_linear":
            grad_embedding = torch.zeros(
                [len(indices), (encoder_dim + 1) * num_targets]
            )
        else:
            raise ValueError(
                f"Grad embedding type '{grad_embedding_type}' not supported."
                "Viable options: 'bias', 'linear', or 'bias_linear'"
            )

        index = 0

        for i, batch in enumerate(iter, 1):
            # print(f"Batch: {i}/{len(iter)}")
            x, lengths = batch.text
            start = index
            end = start + x.shape[0]

            logits, return_dict = model(x, lengths=lengths)
            l1 = return_dict["encoded"]
            if num_targets == 1:
                y_pred = torch.sigmoid(logits)
                y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

            y_pred = logits.max(1)[1]

            if logits.shape[-1] == 1:
                # Binary cross entropy, cast labels to float
                y_pred = y_pred.type(torch.float)

            squeezed_logits = logits.squeeze()
            if len(squeezed_logits.shape) == 0:
                squeezed_logits = squeezed_logits.unsqueeze(dim=0)

            loss = criterion(squeezed_logits, y_pred)
            l0_grads = torch.autograd.grad(loss, logits)[0]

            # Calculate the linear layer gradients if needed.
            if grad_embedding_type != "bias":
                l0_expand = torch.repeat_interleave(l0_grads, encoder_dim, dim=1)
                l1_grads = l0_expand * l1.repeat(1, num_targets)

            # Populate embedding tensor according to the supplied argument.
            if grad_embedding_type == "bias":
                grad_embedding[start:end] = l0_grads
            elif grad_embedding_type == "linear":
                grad_embedding[start:end] = l1_grads
            else:
                grad_embedding[start:end] = torch.cat([l0_grads, l1_grads], dim=1)

            index = end

            # Empty the cache as the gradient embeddings could be very large.
            torch.cuda.empty_cache()

        model.to(self.device)
        return grad_embedding

    def _get_representation_gradient(
        self,
        model,
        indices,
        device,
    ):

        # cpu_device = torch.device("cpu")
        # model.to(cpu_device)
        iter = make_iterable(
            self.dataset, device, batch_size=self.batch_size, indices=indices
        )

        enc = []
        grads = []

        for batch in iter:

            inputs, lengths = batch.text
            inputs.requires_grad = False
            # pad_idx = self.meta.padding_idx
            # attention_mask = inputs != pad_idx
            name = model.get_classifier_name()
            clf = getattr(model.classifier, name)
            config = model.classifier.config
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
            encoded = clf.encoder(
                embedded_tokens,
                # head_mask=head_mask,
                # attention_mask=attention_mask,
            )[0][:, 0]

            enc.append(encoded.cpu())

            mean = encoded.mean()
            mean.backward()
            enc_grad = embedded_tokens.grad.data
            norm = enc_grad.norm(p=2, dim=(1, 2))
            grads.append(norm)
            # mean = torch.mean(norm, 1)
            # mean = mean.repeat_interleave(norm.shape[1]).reshape(-1, norm.shape[1])

            # Empty the cache as the gradient embeddings could be very large.
            torch.cuda.empty_cache()

        model.to(self.device)
        return torch.cat(grads)

    def _get_representation_mean(
        self,
        model,
        indices,
        device,
    ):

        # cpu_device = torch.device("cpu")
        # model.to(cpu_device)
        iter = make_iterable(
            self.dataset, device, batch_size=self.batch_size, indices=indices
        )

        enc = []
        grads = []

        for batch in iter:

            inputs, lengths = batch.text
            inputs.requires_grad = False
            # pad_idx = self.meta.padding_idx
            # attention_mask = inputs != pad_idx
            name = model.get_classifier_name()
            clf = getattr(model.classifier, name)
            config = model.classifier.config
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
            encoded = clf.encoder(
                embedded_tokens,
                # head_mask=head_mask,
                # attention_mask=attention_mask,
            )[0][:, 0]

            enc.append(encoded.cpu())

            mean = encoded.mean()
            mean.backward()
            enc_grad = embedded_tokens.grad.data
            norm = enc_grad.norm(p=2, dim=(1, 2))
            grads.append(norm)
            # mean = torch.mean(norm, 1)
            # mean = mean.repeat_interleave(norm.shape[1]).reshape(-1, norm.shape[1])

            # Empty the cache as the gradient embeddings could be very large.
            torch.cuda.empty_cache()

        enc = torch.cat(enc).cpu()
        dist = torch.cdist(enc, enc)

        return dist.mean(dim=1)


class RandomSampler(Sampler):
    name = "random"

    def query(self, query_size, unlab_inds, **kwargs):
        return np.random.choice(unlab_inds, size=query_size, replace=False)

    def weights(self, query_size, unlab_inds, **kwargs):
        return np.array(1.0).repeat(unlab_inds.size)
