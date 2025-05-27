from typing import Optional, Tuple

import torch

from src.data.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.modules.hf_transformer_base_module import HFTransformerBaseModule


class HFTransformerModule(HFTransformerBaseModule):
    def forward(self, attention_mask, **kwargs):

        outputs = self.encoder(
            input_ids=kwargs["input_ids"].to(self.encoder.device),
            attention_mask=attention_mask.to(self.encoder.device),
        )

        embeddings = outputs.last_hidden_state
        embeddings = self.embedding_post_processor(embeddings)
        return embeddings

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass of the model and calculate the loss if label_data is provided.

        Args:
            model_input: The input data to the model.
            label_data: The label data to the model. Its optional as it is not required for inference.
        """

        model_output = self.forward(
            attention_mask=model_input.mask,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            }
        )

        if label_data is None:
            return model_output, torch.tensor(0.0).to(model_output.device)
        # hardcoded for now, need to change for multiple features:
        # policy
        assert len(label_data.labels) == 1, "Only one label is supported for now"
        for label in label_data.labels:
            curr_label = label_data.labels[label]
            current_label_location = label_data.label_location[label]

        loss = self.criterion(
            query_embeddings=model_output,
            key_embeddings=self.get_embedding_table().to(model_output.device),
            label_locations=current_label_location.to(model_output.device),
            labels=curr_label.to(model_output.device),
        )
        return model_output, loss
