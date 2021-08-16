import numpy as np
import random
import torch
from torch.utils.data import Dataset
from abcd.utils.arguments import Config
from typing import List, Optional, Dict, Any, Sequence, Union

from typing import TypeVar
from abc import ABC, abstractmethod
from .feature_dataclasses import (
    BaseFeature,
    FeatureType,
    ActionFeature,
    CascadeFeature,
    CompletionFeature,
)
from dataclasses import fields


class _BaseFeature(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids: List[int],
        segment_ids: List[int],
        input_mask: List[int],
        label_id: Optional[int],
        position_ids: Optional[List[int]] = None,
    ):
        self.input_id = input_ids
        self.segment_id = segment_ids
        self.mask_id = input_mask
        self.label_id = label_id
        self.position_id = position_ids


class _ActionFeature(_BaseFeature):
    """ A single set of features with precomputed context token ids"""

    def __init__(
        self,
        input_ids: List[int],
        segment_ids: List[int],
        input_mask: List[int],
        label_ids: Dict,
        context: Dict,
    ):
        super().__init__(input_ids, segment_ids, input_mask, label_ids["value"])
        # token_ids is a batch_size length list, where each item is 100 ids
        self.context_token: List[List[int]] = context["token_ids"]
        self.context_segment = context["segment_ids"]
        self.context_mask = context["mask_ids"]
        self.action_id = label_ids["action"]


class _CompletionFeature(_BaseFeature):
    """ A single set of completion features with precomputed context token ids"""

    def __init__(
        self,
        input_ids: List[int],
        segment_ids: List[int],
        input_mask: List[int],
        label_ids: Dict,
        context: Dict,
        candidates: Any,
    ):
        super().__init__(input_ids, segment_ids, input_mask, None)
        self.candidates = candidates
        self.context_token = context["token_ids"]
        self.context_segment = context["segment_ids"]
        self.context_mask = context["mask_ids"]

        self.intent_id = label_ids["intent"]
        self.nextstep_id = label_ids["nextstep"]
        self.action_id = label_ids["action"]
        self.value_id = label_ids["value"]
        self.utt_id = label_ids["utterance"]


CompletionFeatureType = TypeVar(
    "CompletionFeatureType", bound=CompletionFeature, covariant=True
)


class _CascadeFeature(_CompletionFeature):
    """ A single set of completion features with precomputed context token ids"""

    def __init__(
        self,
        input_ids: List[int],
        segment_ids: List[int],
        input_mask: List[int],
        label_ids: Dict,
        context: Dict,
        candidates: Any,
    ):
        super().__init__(
            input_ids, segment_ids, input_mask, label_ids, context, candidates
        )
        self.convo_id = label_ids["convo"]
        self.turn_count = label_ids["turn"]


class BaseDataset(Dataset[FeatureType], ABC):
    def __init__(self, args: Config, features: List[FeatureType]):
        self.data = features
        self.model_type = args.model_type
        self.num_examples = len(features)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> FeatureType:
        return self.data[idx]

    @abstractmethod
    def collate_func(self, features: Sequence[FeatureType]) -> FeatureType:
        # def collate_func(self, args, split, raw_data):
        raise NotImplementedError()


class ActionDataset(BaseDataset[ActionFeature]):
    def collate_func(self, features: Sequence[ActionFeature]) -> ActionFeature:
        return ActionFeature.stack(features)

        # input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
        # segment_ids = torch.tensor([f.segment_id for f in features], dtype=torch.long)
        # mask_ids = torch.tensor([f.mask_id for f in features], dtype=torch.long)
        # context_tokens = torch.tensor(
        #     [f.context_tokens for f in features], dtype=torch.long
        # )
        # context_segments = torch.tensor(
        #     [f.context_segments for f in features], dtype=torch.long
        # )
        # context_masks = torch.tensor(
        #     [f.context_masks for f in features], dtype=torch.long
        # )
        # action_ids = torch.tensor([f.action_id for f in features], dtype=torch.long)
        # value_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        # return ActionFeature(
        #     input_id=input_ids,
        #     segment_id=segment_ids,
        #     input_mask=mask_ids,
        #     context_token=context_tokens,
        #     context_segment=context_segments,
        #     context_mask=context_masks,
        #     action_id=action_ids,
        #     value_id=value_ids,
        # )
        # return (
        #     input_ids,
        #     segment_ids,
        #     mask_ids,
        #     context_tokens,
        #     context_segments,
        #     context_masks,
        #     action_ids,
        #     value_ids,
        # )


class CompletionDataset(BaseDataset[CompletionFeatureType]):
    def collate_func(self, features: Sequence[CompletionFeatureType]) -> CompletionFeatureType:
        return CompletionFeature.stack(features)

        # input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
        # segment_ids = torch.tensor([f.segment_id for f in features], dtype=torch.long)
        # mask_ids = torch.tensor([f.mask_id for f in features], dtype=torch.long)
        # context_tokens = torch.tensor(
        #     [f.context_token for f in features], dtype=torch.long
        # )
        # context_segments = torch.tensor(
        #     [f.context_segment for f in features], dtype=torch.long
        # )
        # context_masks = torch.tensor(
        #     [f.context_mask for f in features], dtype=torch.long
        # )

        # intent_ids = torch.tensor([f.intent_id for f in features], dtype=torch.long)
        # nextstep_ids = torch.tensor([f.nextstep_id for f in features], dtype=torch.long)
        # action_ids = torch.tensor([f.action_id for f in features], dtype=torch.long)
        # value_ids = torch.tensor([f.value_id for f in features], dtype=torch.long)
        # utterance_ids = torch.tensor([f.utt_id for f in features], dtype=torch.long)
        # all_candidates = torch.tensor(
        #     [f.candidates for f in features], dtype=torch.long
        # )

        # return (
        #     input_ids,
        #     segment_ids,
        #     mask_ids,
        #     context_tokens,
        #     context_segments,
        #     context_masks,
        #     intent_ids,
        #     nextstep_ids,
        #     action_ids,
        #     value_ids,
        #     utterance_ids,
        #     all_candidates,
        # )


class CascadeDataset(CompletionDataset[CascadeFeature]):
    def collate_func(self, features: Sequence[CascadeFeature]) -> CascadeFeature:
        return CascadeFeature.stack(features)

        # collated_batch = super().collate_func(features)
        # convo_ids = torch.tensor([f.convo_id for f in features], dtype=torch.long)
        # turn_counts = torch.tensor([f.turn_count for f in features], dtype=torch.long)
        # cascade_batch = (convo_ids, turn_counts)

        # return collated_batch + cascade_batch
