from enum import Enum

import torch
import torch.nn.functional as F


class InBatchContrastiveLoss(torch.nn.Module):
    """Contrastive loss for InBatchPredictionTask
    Parameters:
    -----------
        tau: float
            Temperature parameter for the contrastive loss.
        normalize: bool
            Whether to normalize the embeddings before computing the similarity.
    """

    def __init__(
        self,
        contrastive_tau: float = 0.1,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.tau = contrastive_tau
        self.normalize = normalize

    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        label_locations: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        # query_embeddings: (batch_size x sequence length x embedding_dim)
        # query_embeddings is the output of the transformer
        # key_embeddings: (total number of vocabs or number of masks x embedding_dim)
        # key_embeddings is the embedding from the embedding table
        # label_locations: (number of masks x 2)
        # labels: (number of masks)
        label_locations = label_locations.to(query_embeddings.device)
        labels = labels.to(query_embeddings.device)
        # we conduct in-batch prediction only
        if len(labels) != len(key_embeddings):
            # we enter this loop only when we are not using parameter server
            # key_embeddings with parameter server is always the same length as labels
            # because we previously use label indices to get the embeddings from PS
            key_embeddings = key_embeddings[labels]

        # having this mask cuz we need to avoiding comparing the same item with itself
        mask = labels.expand(labels.shape[0], -1) != labels.reshape(-1, 1)

        # get representation of masked tokens
        # label_locations[:, 0] refers to the index of sequences
        # label_locations[:, 1] refers to the index of tokens in the sequences
        query_embeddings = query_embeddings[
            label_locations[:, 0], label_locations[:, 1]
        ]

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            key_embeddings = F.normalize(key_embeddings, dim=-1)

        # a good reference to this part could be:
        # Equation (1) in https://arxiv.org/abs/2002.05709
        logits = torch.mm(query_embeddings, key_embeddings.t())

        numerator = torch.exp(torch.diagonal(logits) / self.tau)
        denominator = torch.sum(torch.exp(torch.mul(logits, mask) / self.tau), dim=-1)
        loss = -torch.log(numerator / denominator)

        return loss.mean()


class FullBatchCrossEntropyLoss(torch.nn.Module):
    """FullBatchCrossEntropyLoss, it computes the logit for every single candidate in the embedding table
    Parameters:
    -----------
        tau: float
            Temperature parameter for the contrastive loss.
        normalize: bool
            Whether to normalize the embeddings before computing the similarity.
    """

    def __init__(
        self,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        self.cross_entroy_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        label_locations: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        # query_embeddings: (batch_size x sequence length x embedding_dim)
        # query_embeddings is the output of the transformer
        # key_embeddings: (total number of vocabs x embedding_dim)
        # key_embeddings is the embedding from the embedding table
        # label_locations: (number of masks x 2)
        # labels: (number of masks)

        # get representation of masked tokens
        # label_locations[:, 0] refers to the index of sequences
        # label_locations[:, 1] refers to the index of tokens in the sequences
        query_embeddings = query_embeddings[
            label_locations[:, 0], label_locations[:, 1]
        ]

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            key_embeddings = F.normalize(key_embeddings, dim=-1)

        logits = torch.mm(query_embeddings, key_embeddings.t())

        loss = self.cross_entroy_loss(logits, labels.long())

        return loss
