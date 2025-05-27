from abc import ABC, abstractmethod

import torch


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        pass


class MeanAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = torch.cumsum(embeddings, dim=1)
        return embeddings[row_ids, last_item_index, :].squeeze() / (
            last_item_index + 1
        ).reshape(-1, 1)


class LastAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        return embeddings[row_ids, last_item_index]
