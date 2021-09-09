import math
import os
import pdb
import random
import sys
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union, overload

import numpy as np
import torch
from abcd.components.feature_dataclasses import (
    ActionFeature,
    BaseFeature,
    CascadeFeature,
)
from abcd.utils.arguments import Config
from torch import Tensor

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict  # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(args: Config):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup_gpus(args: Config) -> Config:
    n_gpu = 0  # set the default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    # NOTE: Replacing this next line with the property on the Config dataclass:
    # args.n_gpu = n_gpu
    # Therefore we just check that the value would have been the same anyway.
    assert args.n_gpu == n_gpu

    if n_gpu > 0:  # this is not an 'else' statement and cannot be combined
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.debug:
        args.epochs = 3
    return args


def check_cache(args: Config, cache_dir: str) -> Tuple[str, bool]:
    cache_filename = f"{args.model_type}_{args.task}"
    if args.cascade:
        cache_filename += "_cascade"
    if args.use_intent:
        cache_filename += "_intent"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"Loading features from cached file {cache_path}")
        return cache_path, True
    else:
        print(f"Loading raw data and preparing new features")
        return cache_path, False


def check_directories(args: Config):
    cache_dir = os.path.join(args.input_dir, "cache")
    checkpoint_folder = f"{args.prefix}_{args.filename}_{args.model_type}_{args.suffix}"
    ckpt_dir = os.path.join(args.output_dir, args.task, checkpoint_folder)
    directories = [args.input_dir, cache_dir, args.output_dir, ckpt_dir]

    for directory in directories:
        if os.path.exists(directory):
            if directory == ckpt_dir:
                print(f"Warning: {directory} exists and files may be overwritten")
        else:
            print(f"Creating {directory} directory ...")
            os.makedirs(directory)

    cache_results = check_cache(args, cache_dir)
    return ckpt_dir, cache_results


# Define some types for the return value of the `prepare_inputs` function below:

class ModelInputDict(TypedDict):
    """ TypedDict for the formatted input to the model. """
    input_ids: Tensor
    token_type_ids: Tensor
    attention_mask: Tensor


class ASTTargetsTuple(NamedTuple):
    """ NamedTuple for the 'targets'/'labels' of a model in the AST task.

    NOTE: AST: Action State Tracking : Classify the aciton and the value taken at a
    given step.
    """
    action_id: Tensor
    value_id: Tensor


class CDSTargetsTuple(NamedTuple):
    """ NamedTuple for the 'targets'/labels for a model in the CDS task.

    NOTE: CDS: Cascading Dialogue Success: Given the context, predict the rest of the
    conversation. (TODO: @lebrice not 100% sure about this)
    """
    #           intent   nextstep   action    value     utterance
    # targets = [batch[6], batch[7], batch[8], batch[9], batch[10]]
    intent_id: Tensor
    nextstep_id: Tensor
    action_id: Tensor
    value_id: Tensor
    utt_id: Tensor
    convo_id: Optional[Union[int, Tensor]] = None
    turn_count: Optional[Union[int, Tensor]] = None


@overload
def prepare_inputs(
    args: Config, batch: CascadeFeature, speaker_turn: bool
) -> Tuple[ModelInputDict, CDSTargetsTuple, ModelInputDict, Any]:
    ...

# (BUG: This is fine, but mypy doesn't like it, so putting a type: ignore for now).
@overload
def prepare_inputs(  # type: ignore 
    args: Config, batch: ActionFeature, speaker_turn: bool
) -> Tuple[ModelInputDict, ASTTargetsTuple, ModelInputDict, Any]:
    ...

def prepare_inputs(
    args: Config,
    batch: Union[ActionFeature, CascadeFeature],
    speaker_turn: bool = False,
) -> Tuple[ModelInputDict, Union[ASTTargetsTuple, CDSTargetsTuple], ModelInputDict, Any]:
    """
    Convert the `Feature` object into what the transformer models expect as an input for
    the given task.
    """
    if args.task == "ast":
        assert isinstance(batch, ActionFeature)
        full_history: ModelInputDict = {
            # "input_ids": batch[0],
            # "token_type_ids": batch[1],
            # "attention_mask": batch[2],
            "input_ids": batch.input_ids,
            "token_type_ids": batch.segment_ids,
            "attention_mask": batch.input_mask,
        }
        context_tokens: ModelInputDict = {
            # "input_ids": batch[3],
            # "token_type_ids": batch[4],
            # "attention_mask": batch[5],
            "input_ids": batch.context_tokens,
            "token_type_ids": batch.context_segments,
            "attention_mask": batch.context_masks,
        }
        # targets = [batch[6], batch[7]]  # actions and values
        ast_targets = ASTTargetsTuple(
            action_id=batch.action_id, value_id=batch.label_id
        )
        tools: Any = device
        return full_history, ast_targets, context_tokens, tools

    assert isinstance(batch, CascadeFeature)
    full_history = {
        "input_ids": batch.input_ids,
        "token_type_ids": batch.segment_ids,
        "attention_mask": batch.input_mask,
        # "input_ids": batch[0],
        # "token_type_ids": batch[1],
        # "attention_mask": batch[2],
    }
    context_tokens = {
        "input_ids": batch.context_token,
        "token_type_ids": batch.context_segment,
        "attention_mask": batch.context_mask,
        # "input_ids": batch[3],
        # "token_type_ids": batch[4],
        # "attention_mask": batch[5],
    }

    # candidates = batch[11]
    candidates = batch.candidates

    #           intent   nextstep   action    value     utterance
    # targets = [batch[6], batch[7], batch[8], batch[9], batch[10]]
    cds_targets = CDSTargetsTuple(
        intent_id=batch.intent_id,
        nextstep_id=batch.nextstep_id,
        action_id=batch.action_id,
        value_id=batch.value_id,
        utt_id=batch.utt_id,
        convo_id=batch.convo_id if args.cascade else None,
        turn_count=batch.turn_count if args.cascade else None,
    )

    # NOTE: @lebrice: Will use a 'targets' tuple with None values when `args.cascade` is
    # False, rather than use a tuple with more items when it is.
    # Will need to check that there isn't a switch somewhere that is based on the length
    # of the tuple though.
    # if args.cascade:
    #     # targets.append(batch[15])  # convo_ids
    #     # targets.append(batch[16])  # turn_counts
    #     targets.append(batch.convo_id)  # convo_ids
    #     targets.append(batch.turn_count)  # turn_counts
    if args.use_intent:
        tools = candidates, device, batch.intent_id
        # tools = candidates, device, batch[6]
    else:
        tools = candidates, device

    return full_history, cds_targets, context_tokens, tools
