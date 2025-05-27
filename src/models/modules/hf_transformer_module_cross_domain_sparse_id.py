import copy
from typing import Any, Dict, List

import torch
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from src.models.modules.hf_transformer_module import HFTransformerModule
from src.utils.utils import delete_module, find_module_shape, reset_parameters


class HFCDSIDTransformerModule(HFTransformerModule):
    def __init__(
        self,
        domain_models: Dict[str, transformers.PreTrainedModel],
        masking_token: int,
        **kwargs,
    ) -> None:
        """Initialize the HFCDSIDTransformerModule.
        This module conducts cross-domain communication through domain-specific encoders.
        During the encoding phase, tokens in each domain can only attend to themselves.
        Tokens from different domains are merged by in-place addition and normalized by a layer norm layer.
        If a shared encoder is provided, the merged embeddings are passed through the shared encoder for further mix up.

        :param domain_model_spec: the model architecture for domain-specific encoders.
        :return None:
        """
        super().__init__(**kwargs)

        embedding_table_dim = find_module_shape(self.encoder, "embed_tokens")
        num_embeddings, embedding_dim = embedding_table_dim
        self.embedding_table = torch.nn.Embedding(
            num_embeddings=num_embeddings,  # type: ignore
            embedding_dim=embedding_dim,  # type: ignore
        )

        # deleting embedding table in the encoder
        delete_module(self.encoder, "embed_tokens")
        delete_module(self.encoder, "shared")

        self.masking_token = masking_token

        # creating domain encoders
        self.domain_encoders = torch.nn.ModuleDict(
            {  # converting to string because torch enforces keys to be strings
                domain: self._spawn_domain_encoder_as_module_list(
                    domain_encoder=domain_models[domain],
                )
                for domain in domain_models
            }
        )
        reset_parameters(self.encoder)
        self.embedding_table.weight.data.uniform_(-0.05, 0.05)
        self.domain_fusion_layer_norm = torch.nn.LayerNorm(embedding_dim)

        # removing general encoder if it does not have any layer
        if self.encoder.config.num_layers == 0:
            self.encoder = None

    def forward(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        embeddings = self.embedding_table(
            input_ids.to(self.embedding_table.weight.device)
        )
        domain_masks = self.create_domain_mask(
            input_ids=input_ids,
            domain_ids=domain_ids,
        )

        domain_specific_embeddings = {}
        for domain, layers in self.domain_encoders.items():
            hidden_embeds = embeddings
            for layer in layers:
                hidden_embeds = layer(
                    inputs_embeds=hidden_embeds,
                    attention_mask=domain_masks[domain],
                ).last_hidden_state
            domain_specific_embeddings[domain] = hidden_embeds

        # merging domain embeddings
        embeddings = self.merge_domain_embeddings(
            embeddings=embeddings,
            domain_specific_embeddings=domain_specific_embeddings,
            domain_masks=domain_masks,
            attention_mask=attention_mask,
        )

        embeddings = self.embedding_post_processor(embeddings)

        return embeddings

    def _spawn_domain_encoder_as_module_list(
        self,
        domain_encoder: transformers.PreTrainedModel,
    ) -> torch.nn.ModuleList:
        """Duplicate a encoder as a modulelist, where each layer is a 1-layer encoder
        and re-initialze the duplicated model's parameter.

        :param domain_encoder: the encoder to be duplicated.
        :return: the duplicated and re-initialized encoder.
        """
        num_layers = domain_encoder.config.num_layers

        domain_encoder.config.num_layers = (
            1  # setting this to because we want an instance of just 1 layer
        )
        # multi-layer transformer can be achieve by a for loop using module list
        # as shown below

        # instantiating a single layer encoder
        domain_encoder_single_layer = domain_encoder.__class__(domain_encoder.config)

        # removing dummy embedding table in the one-layer transformer
        delete_module(domain_encoder_single_layer, "embed_tokens")
        delete_module(domain_encoder_single_layer, "shared")

        domain_encoder = torch.nn.ModuleList()

        # building a multi-layer transformer as a module list instead of a single class
        # doing this because it will have give us great freedom to manipulate latent between layers
        # which is necessary for iterating cross-domain research ideas
        for _ in range(num_layers):
            reset_parameters(domain_encoder_single_layer)
            domain_encoder.append(copy.deepcopy(domain_encoder_single_layer))
        return domain_encoder

    def get_embedding_table(self) -> torch.Tensor:
        """overriding the get_embedding_table method.
        because the embedding table is different from the encoder's embedding table.
        """
        if self.hparams.weight_tying:  # type: ignore
            return self.embedding_table.weight
        else:
            return self.decoder.weight

    def merge_domain_embeddings(
        self,
        embeddings: torch.Tensor,
        domain_specific_embeddings: Dict[str, torch.Tensor],
        domain_masks: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Combine domain-specific embeddings returned by their encoders
        embeddings are merged by in-place addition according to their domain masks and normalized by a layer norm layer.
        if a shared encoder is provided, the merged embeddings are passed through the shared encoder for futher mix up.

        :param embeddings:
            shape: [bsz x seq_len x hid_dim]
            the embeddings from the embedding table.
        :param domain_specific_embeddings:
            shape: [bsz x seq_len x hid_dim]
            the domain-specific embeddings from domain-specific encoder.
            the same size as embeddings as
                1. for parallel computation
                2. out-of-domain tokens will not be included in the computation
        :param domain_masks: the domain mask for each domain.
        :param attention_mask: the attention mask for all content.
        :return torch.Tensor: merged sequences.
        """
        # in-place update of embeddings with domain_specific embeddings
        for domain in self.domain_encoders:
            mask = domain_masks[domain]
            embeddings[mask] += domain_specific_embeddings[domain][mask]

        # we can prob explore other merging schemes here
        embeddings = self.domain_fusion_layer_norm(embeddings)

        # whether or not we should fuse the domain embeddings with another transformer
        # (i.e., the general encoder)
        if self.encoder:
            outputs: BaseModelOutputWithPastAndCrossAttentions = self.encoder(
                inputs_embeds=embeddings,
                attention_mask=attention_mask.to(self.embedding_table.weight.device),
            )

            embeddings = outputs.last_hidden_state
        return embeddings

    def create_domain_mask(
        self,
        input_ids: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Creating domain mask indicating if a token belongs to a domain.
        :param input_ids: the raw sparse ids.
        :param domain_ids: raw domain ids indicating which domain a input id belongs to.
        :return Dict[torch.Tensor]: a dict containing domain-domain_mask pairs
        """
        domain_masks = {}
        for domain in self.domain_encoders:
            # creating domain mask: a token is true if
            # 1: it is a masking token
            # or 2: it belongs to the domain
            # the high-level idea is that we dont know the domain of the masking token
            # so we include it in all domains
            domain_mask = torch.logical_or(
                input_ids == self.masking_token,
                domain_ids == int(domain),
            )
            domain_masks[domain] = domain_mask.to(self.embedding_table.weight.device)
        return domain_masks


class HFCDSIDIBTransformerModule(HFCDSIDTransformerModule):
    def __init__(
        self,
        num_ib_tokens: int,
        num_placeholder_ids: int,
        ib_comm_layers: List[int],
        **kwargs,
    ) -> None:
        """Initialize the HFCDSIDIBTransformerModule.
        This module conducts cross-domain communication through information bottleneck (IB) tokens.
        tokens in each domain can all attend to themselves and IB tokens,
        and IB tokens are updated by combining of all domain-specific IB embeddings.
        This idea is inspired by the work of Attention Bottlenecks for Multimodal Fusion
        https://proceedings.neurips.cc/paper/2021/file/76ba9f564ebbc35b1014ac498fafadd0-Paper.pdf

        :param num_ib_tokens: the number of IB tokens, if ib_tokens  == 0, this module is equivalent to HFCDSIDTransformerModule.
        :param num_placeholder_ids: the number of total reserved placeholder ids, used as a sanity check to determine
                                    if IB token indices make sense.
        :param ib_comm_layers: indices of layers where IB tokens are used for communication
        :return None:
        """
        super().__init__(**kwargs)
        assert (
            num_placeholder_ids > num_ib_tokens
        ), "IB token index must not exceed the total number of placeholder ids"
        self.num_ib_tokens = num_ib_tokens
        self.ib_comm_layers = ib_comm_layers

    def ib_token_sync(
        self,
        domain_embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """aggregate IB embeddings from all domains and update all domain embeddings with the aggregated IB embeddings.
        IB embeddings are aggregated by mean pooling and replaced by in-place operation.

        :param domain_embeddings: the domain-specific embeddings from domain-specific encoder.
        :return  Dict[torch.Tensor]: a dict containing domain-domain_mask pairs
        """
        # initialize IB latent as None
        aggregated_ib_embeddings = None

        # aggregating IB embeddings from all domains
        for domain in domain_embeddings:
            update_term = domain_embeddings[domain][:, : self.num_ib_tokens]
            if aggregated_ib_embeddings is None:
                aggregated_ib_embeddings = torch.zeros_like(update_term)
            aggregated_ib_embeddings += update_term

        aggregated_ib_embeddings /= len(domain_embeddings)  # mean pooling

        # updating the IB embeddings in all domains
        for domain in domain_embeddings:
            domain_embeddings[domain][
                :, : self.num_ib_tokens
            ] = aggregated_ib_embeddings

        return domain_embeddings

    def cross_domain_ib_forward(
        self,
        inputs_embeds: torch.Tensor,
        domain_masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for cross-domain communication through IB tokens.

        :param inputs_embeds: embeddings from the embedding table.
        :param domain_masks: boolean domain indicators.
        :return  Dict[torch.Tensor]: a dict containing domain-domain_mask pairs
        """
        # deep copying this mask s.t. later in-place operation does not affect the original mask
        domain_masks = copy.deepcopy(domain_masks)
        domain_embeddings = {}

        # creating IB embeddings, by default the IB token start from 2
        # 0 and 1 are reserved for padding and masking tokens, and we have 0-99 being placeholder tokens
        # so 2-99 token ids are free
        ib_embedding: torch.Tensor = self.embedding_table(
            torch.arange(
                2,
                2 + self.num_ib_tokens,
            ).to(self.embedding_table.weight.device)
        )
        # expanding to match the batch size
        ib_embedding = ib_embedding.unsqueeze(0).repeat(len(inputs_embeds), 1, 1)
        # after the concatenation, each sequence looks like :
        # [IB_TOKEN_0, IB_TOKEN_1...,IB_TOKEN_K, REAL_TOKEN_0, REAL_TOKEN_1, ..., REAL_TOKEN_N, PADDING_TOKENS...]
        embeddings = torch.cat([ib_embedding, inputs_embeds], dim=1)

        for domain, domain_mask in domain_masks.items():
            # for each domain, we modify the domain mask to include the IB tokens
            # so that tokens in each domain can all attend to IB tokens
            domain_masks[domain] = torch.cat(
                [
                    torch.ones(
                        len(inputs_embeds), self.num_ib_tokens, dtype=torch.bool
                    ).to(self.embedding_table.weight.device),
                    domain_mask,
                ],
                dim=1,
            )
            # initialize the domain embeddings
            domain_embeddings[domain] = embeddings

        self.attention_score = 0
        # looping over all layers in domain-specific transformers
        for layer_index in range(
            len(self.domain_encoders[next(iter(self.domain_encoders))])
        ):
            # we do communication across all domains at each layer through IB tokens
            for domain in self.domain_encoders:
                # communication within each domain
                output = self.domain_encoders[domain][layer_index](
                    inputs_embeds=domain_embeddings[domain],
                    attention_mask=domain_masks[domain],
                    output_attentions=True,
                )
                self.attention_score += torch.stack(
                    [
                        head_attention[
                            :, :, self.num_ib_tokens :, : self.num_ib_tokens
                        ].mean()
                        for head_attention in output.attentions
                    ]
                ).mean()
                hidden_embeds_this_domain = output.last_hidden_state
                domain_embeddings[domain] = hidden_embeds_this_domain
            # communication across domains through IB tokens
            if layer_index in self.ib_comm_layers:
                domain_embeddings = self.ib_token_sync(domain_embeddings)

        for domain in self.domain_encoders:
            # IB tokens are discarded from the domain embeddings
            # as they are not needed for the final output
            domain_embeddings[domain] = domain_embeddings[domain][
                :, self.num_ib_tokens :
            ]
        return domain_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        if self.num_ib_tokens == 0:
            return super().forward(
                input_ids=input_ids,
                domain_ids=domain_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        embeddings = self.embedding_table(
            input_ids.to(self.embedding_table.weight.device)
        )
        domain_masks = self.create_domain_mask(
            input_ids=input_ids,
            domain_ids=domain_ids,
        )

        domain_specific_embeddings = self.cross_domain_ib_forward(
            inputs_embeds=embeddings,
            domain_masks=domain_masks,
        )
        # in-place update of embeddings with domain_specific embeddings
        embeddings = self.merge_domain_embeddings(
            embeddings=embeddings,
            domain_specific_embeddings=domain_specific_embeddings,
            domain_masks=domain_masks,
            attention_mask=attention_mask,
        )

        embeddings = self.embedding_post_processor(embeddings)

        return embeddings
