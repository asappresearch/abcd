from dataclasses import dataclass, field, InitVar, astuple, asdict, fields
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union,
    Tuple,
    Iterator,
    Iterable,
    Sequence,
    TypeVar,
)
from simple_parsing.helpers.serialization import JsonSerializable
from torch import Tensor
import torch
import numpy as np


@dataclass
class BaseFeature(JsonSerializable):
    input_ids: List[int]
    segment_ids: List[int]
    input_mask: List[int]

    @property
    def mask_id(self) -> List[int]:
        return self.input_mask

    def keys(self):
        return tuple(k for k, _ in self.items())

    def values(self):
        return tuple(v for _, v in self.items())

    def items(self) -> Iterable[Tuple[str, Any]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def _as_dict(self) -> Dict:
        return asdict(self)

    def _as_tuple(self) -> Tuple:
        return astuple(self)

    def __iter__(self) -> Iterator[Tuple[Tensor]]:
        yield from self._as_tuple()

    def to(self, device: Union[str, torch.device]):
        return type(self)(**{key: value.to(device) for key, value in self.items()})

    @classmethod
    def stack(cls, features: Sequence["FeatureType"]) -> "FeatureType":
        kwargs = {
            field.name: torch.tensor(
                [getattr(feat, field.name) for feat in features], dtype=torch.long
            )
            for field in fields(cls)
        }
        stacked = cls(**kwargs)
        # assert False, [v.shape for v in stacked.values()]
        return stacked


FeatureType = TypeVar("FeatureType", bound=BaseFeature, covariant=True)


@dataclass
class ActionFeature(BaseFeature):

    context: InitVar[Dict] = {}
    context_tokens: List[int] = field(default_factory=list)  # = context["token_ids"]
    context_segments: List[int] = field(
        default_factory=list
    )  # = context["segment_ids"]
    context_masks: List[int] = field(default_factory=list)  # = context["mask_ids"]

    label_ids: InitVar[Dict[str, int]] = {}
    action_id: int = field(default=None)  # = label_ids["action"]
    label_id: int = field(default=None)  # = label_ids["value"]

    def __post_init__(self, context: Dict, label_ids: Dict[str, int]):
        if context or label_ids:
            if "token_ids" in label_ids:
                # BUG: These get switched for some reason?
                context, label_ids = label_ids, context
            assert "token_ids" in context, (context, label_ids)
            self.context_tokens = context["token_ids"]
            self.context_segments = context["segment_ids"]
            self.context_masks = context["mask_ids"]
            self.action_id = label_ids["action"]
            self.label_id = label_ids["value"]


@dataclass
class CompletionFeature(BaseFeature):
    candidates: List

    context: InitVar[Dict] = {}
    label_ids: InitVar[Dict] = {}

    context_token: List[List[int]] = field(default_factory=list)
    context_segment: List[int] = field(default_factory=list)
    context_mask: List[int] = field(default_factory=list)

    intent_id: Union[int, List[int]] = field(default_factory=list)
    nextstep_id: Union[int, List[int]] = field(default_factory=list)
    action_id: Union[int, List[int]] = field(default_factory=list)
    value_id: Union[int, List[int]] = field(default_factory=list)
    utt_id: Union[int, List[int]] = field(default_factory=list)

    def __post_init__(self, context: Dict, label_ids: Dict):
        if context or label_ids:
            self.context_token = context["token_ids"]
            self.context_segment = context["segment_ids"]
            self.context_mask = context["mask_ids"]

            self.intent_id = label_ids["intent"]
            self.nextstep_id = label_ids["nextstep"]
            self.action_id = label_ids["action"]
            self.value_id = label_ids["value"]
            self.utt_id = label_ids["utterance"]


@dataclass
class CascadeFeature(CompletionFeature):
    """ A single set of completion features with precomputed context token ids"""

    convo_id: Union[int, List[int]] = field(default_factory=list)
    turn_count: Union[int, List[int]] = field(default_factory=list)

    def __post_init__(self, context: Dict, label_ids: Dict):
        super().__post_init__(context=context, label_ids=label_ids)
        if context or label_ids:
            self.convo_id = label_ids["convo"]
            self.turn_count = label_ids["turn"]


### Examples from `features.py`.


@dataclass
class InputExample(JsonSerializable):
    """A single training/test example for simple sequence classification."""

    # Unique id for the example.
    guid: int
    # list of strings. The untokenized text of the converation so far.
    input_context: List[str]
    # The label of the example. This should be specified for train and dev examples, but
    # not for test examples.
    target_label: Optional[str]
    # list of candidates to choose from for utterance ranking
    candidates: Optional[List[str]] = field(init=False, default=None)

    @property
    def context(self) -> List[str]:
        return self.input_context

    @property
    def label(self) -> Optional[str]:
        return self.target_label


@dataclass
class ActionExample(InputExample):
    """A single training/test example for slot value filling. """

    tokens: List[int]
    action: int

    @property
    def context_tokens(self):
        return self.tokens


@dataclass
class CompleteExample(InputExample):
    """A single training/test example for task completion. """

    tokens: List[int]

    targets: InitVar[Tuple[Any, Any, Any, Any, Any]]

    def __post_init__(self, targets: Tuple[Any, Any, Any, Any, Any]):
        intent, nextstep, action, value_index, utt_index = targets
        self.intent_label = intent
        self.nextstep_label = nextstep
        self.action_label = action
        self.value_label = value_index
        self.utt_label = utt_index

    @property
    def context_tokens(self):
        return self.tokens


class CascadingExample(CompleteExample):
    """A single training/test example for task completion. """

    convo_id: int = field(init=False)
    turn_count: int = field(init=False)

    def __post_init__(self, targets):
        super().__post_init__(targets=targets)

        self.convo_id = convo_id
        self.turn_count = turn_count
