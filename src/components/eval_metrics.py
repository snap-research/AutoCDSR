from typing import Any, Dict, List

import torch
from torchmetrics.retrieval.base import RetrievalMetric


class Evaluator:
    """Wrapper for evaluation metrics.
        It takes model outputs and automatically calculates the ranking metrics.

    Parameters
    ----------
    metrics: List[MetricClass]
        List of metric classes to calculate
    """

    _DEFAULT_METRIC_INIT_VALUE = 0.0
    _DEFAULT_TOTAL_SAMPLE_NAME = "total_sample"

    def __init__(
        self,
        metrics: Dict[str, RetrievalMetric],
        top_k_list: List[int],
        should_sample_negatives_from_vocab: bool = True,
        num_negatives: int = 500,
        placeholder_token_buffer: int = 100,
    ):
        self.metrics = {
            f"{metric_name}@{top_k}": metric_object(
                top_k=top_k, sync_on_compute=True, compute_with_cache=False
            )
            for metric_name, metric_object in metrics.items()
            for top_k in top_k_list
        }
        self.should_sample_negatives_from_vocab = should_sample_negatives_from_vocab
        self.num_negatives = num_negatives
        self.placeholder_token_buffer = placeholder_token_buffer

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):

        results = {
            metric_name: Evaluator._DEFAULT_METRIC_INIT_VALUE
            for metric_name in self.metrics.keys()
        }
        results[
            Evaluator._DEFAULT_TOTAL_SAMPLE_NAME
        ] = Evaluator._DEFAULT_METRIC_INIT_VALUE
        num_of_samples = query_embeddings.shape[0]
        num_of_candidates = key_embeddings.shape[0]

        if self.should_sample_negatives_from_vocab:
            inbatch_negatives = self.sample_negative_ids_from_vocab(
                num_of_samples=num_of_samples,
                num_of_candidates=num_of_candidates,
                num_negatives=self.num_negatives,
            )
            # we +1 here because we need to include the positive sample
            num_of_candidates = self.num_negatives + 1
            pos_embeddings = key_embeddings[labels]
            key_embeddings = key_embeddings[inbatch_negatives]
            # key_embeddings shape: (bsz, num_negatives+1, emb_dim)
            key_embeddings = torch.cat(
                [pos_embeddings.unsqueeze(1), key_embeddings], dim=1
            )
            # the positive index will always be 0 because the pos embedding will always be the first one.
            labels = torch.zeros(num_of_samples).long()

        # following examples from https://lightning.ai/docs/torchmetrics/stable/retrieval/precision.html
        # indexes refers to the mask of the labels
        indexes = torch.arange(0, query_embeddings.shape[0])
        expanded_indexes = (
            indexes.unsqueeze(-1).expand(num_of_samples, num_of_candidates).reshape(-1)
        )

        if self.should_sample_negatives_from_vocab:
            preds = (
                torch.mul(
                    query_embeddings.unsqueeze(1).expand_as(key_embeddings),
                    key_embeddings,
                )
                .sum(-1)
                .reshape(-1)
            )
        else:
            preds = torch.mm(query_embeddings, key_embeddings.t()).reshape(-1)

        target = torch.zeros(num_of_samples, num_of_candidates).bool()
        target[torch.arange(num_of_samples), labels] = True
        target = target.reshape(-1)

        results[Evaluator._DEFAULT_TOTAL_SAMPLE_NAME] += num_of_samples

        for metric_name, metric_object in self.metrics.items():
            results[metric_name] += (
                metric_object(
                    preds,
                    target.to(preds.device),
                    indexes=expanded_indexes.to(preds.device),
                )
            ).item()

        # do this to clear the cache, otherwise each preds, target, and indexes will be stored in the retrieval metrics.
        self.reset()

        return results

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    # this method samples random negative samples from the whole vocab
    def sample_negative_ids_from_vocab(
        self,
        num_of_samples: int,
        num_of_candidates: int,
        num_negatives: int,
    ) -> torch.Tensor:
        # num_of_samples: batch size
        # num_of_candidates: number of total vocabs
        # num_negatives: number of negative samples

        # we do randint to accelerate the negative sampling
        # this could have collision with positive pairs but the chance is very low

        negative_candidates = torch.randint(
            self.placeholder_token_buffer,
            num_of_candidates,
            (num_of_samples, num_negatives),
        )

        return negative_candidates


class NDCG(RetrievalMetric):  #
    def __init__(
        self,
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:
        """Check shape, check and convert dtypes, flatten and add to accumulators."""

        bsz = int(len(indexes) / (indexes == 0).sum().item())

        preds = preds.reshape(bsz, -1)
        target = target.reshape(bsz, -1).int()

        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        # Compute DCG
        dcg = torch.sum(
            topk_true
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Compute IDCG
        ideal_indices = torch.topk(target, self.top_k)[1]
        ideal_dcg = torch.sum(
            target.gather(1, ideal_indices)
            / torch.log2(
                torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0)
            ),
            dim=1,
        )

        # Handle cases where IDCG is zero
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)
        return ndcg.mean()

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return None


class Recall(RetrievalMetric):
    def __init__(
        self,
        top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs
    ) -> None:
        """Check shape, check and convert dtypes, flatten and add to accumulators."""

        bsz = int(len(indexes) / (indexes == 0).sum().item())

        preds = preds.reshape(bsz, -1)
        target = target.reshape(bsz, -1).int()

        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        true_positives = topk_true.sum(dim=1)
        total_relevant = target.sum(dim=1)

        recall = true_positives / total_relevant

        return recall.mean()

    def _metric(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return None
