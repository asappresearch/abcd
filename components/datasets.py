import logging
import os, pdb, sys
import random
import math
import torch
import numpy as np
import pickle as pkl

from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm as progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

logger = logging.getLogger(__name__)
try:
  set_start_method('spawn')
except RuntimeError:
  pass

class BaseFeature(object):
  """A single set of features of data."""
  def __init__(self, input_ids, segment_ids, input_mask, label_id, position_ids=None):
    self.input_id = input_ids
    self.segment_id = segment_ids
    self.mask_id = input_mask
    self.label_id = label_id
    self.position_id = position_ids

class ActionFeature(BaseFeature):
  """ A single set of features with precomputed context token ids"""
  def __init__(self, input_ids, segment_ids, input_mask, label_id, 
        context, action_id, position_ids=None):
    super().__init__(input_ids, segment_ids, input_mask, label_id, position_ids)
    # token_ids is a batch_size length list, where each item is 100 ids
    self.context_token = context['token_ids']
    self.context_segment = context['segment_ids']
    self.context_mask = context['mask_ids']
    self.action_id = action_id

class CompletionFeature(BaseFeature):
  """ A single set of completion features with precomputed context token ids"""
  def __init__(self, input_ids, segment_ids, input_mask, label_ids, candidates, context):
    super().__init__(input_ids, segment_ids, input_mask, None)
    self.candidates = candidates
    self.context_token = context['token_ids']
    self.context_segment = context['segment_ids']
    self.context_mask = context['mask_ids']

    self.intent_id = label_ids['intent']
    self.nextstep_id = label_ids['nextstep']
    self.action_id = label_ids['action']
    self.value_id = label_ids['value']
    self.utt_id = label_ids['utterance']

    self.action_mask = int(label_ids['nextstep'] == 1)
    self.value_mask = int(label_ids['value'] >= 0) 
    self.utt_mask = int(label_ids['nextstep'] == 0)

class CascadeFeature(CompletionFeature):
  """ A single set of completion features with precomputed context token ids"""
  def __init__(self, input_ids, segment_ids, input_mask, label_ids, 
          candidates, context, convo_id, turn_count):
    super().__init__(input_ids, segment_ids, input_mask, labels_ids, candidates, context)
    self.convo_id = convo_id
    self.turn_count = turn_count

class BaseDataset(Dataset):

  def __init__(self, args, features, tokenizer):
    self.data = data
    self.tokenizer = tokenizer
    self.ontology = targets
    self.split = split

    self.model_type = args.model_type
    self.include_speaker_turn = args.speaker_turn
    self.cache_dir = args.cache_dir
    self.reprocess = args.reprocess

    self.task_name = loader.name
    self.labels = getattr(loader, f'{self.task_name}_list')
    self.num_examples = len(loader)

    self.tokenizer = tokenizer
    self.num_labels = len(self.labels)
    self.chunk_count = {'train': 1, 'dev': 1, 'test': 1}

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  def collate_func(self, args, split, raw_data):
    raise NotImplementedError()

class ActionDataset(BaseDataset):

  def __init__(self, args, features, split="train"):
    self.data = features
    self.model_type = args.model_type
    self.num_examples = len(features)
    self.split = split

  def collate_func(self, args, split, raw_data):
    examples = []
    for i, line in enumerate(raw_data[split]):
      examples.append(ActionExample(f'{split}_{i}', *line))
    random.shuffle(examples)

    features = self.examples_to_features(examples, self.action_labels, args.max_seq_len,
      self.tokenizer, 'filling', sequence_a_segment_id=0 if self.model_type in ['roberta', 'large'] else 1)
    if len(features) % args.batch_size == 1:
      features = features[:-1]
    
    all_input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_id for f in features], dtype=torch.long)
    all_mask_ids = torch.tensor([f.mask_id for f in features], dtype=torch.long)
    all_context_tokens = torch.tensor([f.context_token for f in features], dtype=torch.long)
    all_context_segments = torch.tensor([f.context_segment for f in features], dtype=torch.long)
    all_context_masks = torch.tensor([f.context_mask for f in features], dtype=torch.long)
    
    all_action_labels = torch.tensor([f.action_id for f in features], dtype=torch.long)
    all_input_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_segment_ids, all_mask_ids, 
              all_context_tokens, all_context_segments, all_context_masks, 
              all_action_labels, all_input_labels)

class CompletionDataset(BaseDataset):

  def __init__(self, args, features, tokenizer):
    loader.tcom_list = []
    super().__init__(args, loader, tokenizer)

    self.intent_labels = loader.intent_list
    self.nextstep_labels = loader.nextstep_list
    self.action_labels = loader.action_list
    self.value_labels = loader.value_list
    self.num_labels = [len(self.intent_labels), len(self.nextstep_labels), 
                        len(self.action_labels), len(self.value_labels), 100]

  def collate_func(self, args, split, raw_data):
    examples = []
    for i, line in enumerate(raw_data[split]):
      examples.append(CompleteExample(f'{split}_{i}', *line))
    random.shuffle(examples)

    labels = [self.intent_labels, self.nextstep_labels, self.action_labels]
    features = self.examples_to_features(examples, labels, args.max_seq_len,
      self.tokenizer, 'completion', sequence_a_segment_id=0 if self.model_type in ['roberta', 'large'] else 1)
    
    all_input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_id for f in features], dtype=torch.long)
    all_mask_ids = torch.tensor([f.mask_id for f in features], dtype=torch.long)

    all_candidates = torch.tensor([f.candidates for f in features], dtype=torch.long)
    all_context_tokens = torch.tensor([f.context_token for f in features], dtype=torch.long)
    all_context_segments = torch.tensor([f.context_segment for f in features], dtype=torch.long)
    all_context_masks = torch.tensor([f.context_mask for f in features], dtype=torch.long)
    
    all_intent_labels = torch.tensor([f.intent_id for f in features], dtype=torch.long)
    all_nextstep_labels = torch.tensor([f.nextstep_id for f in features], dtype=torch.long)
    all_action_labels = torch.tensor([f.action_id for f in features], dtype=torch.long)
    all_value_labels = torch.tensor([f.value_id for f in features], dtype=torch.long)
    all_utterance_labels = torch.tensor([f.utt_id for f in features], dtype=torch.long)

    all_action_masks = torch.tensor([f.action_mask for f in features], dtype=torch.long)
    all_value_masks = torch.tensor([f.value_mask for f in features], dtype=torch.long)
    all_utterance_masks = torch.tensor([f.utt_mask for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_segment_ids, all_mask_ids, all_candidates,
              all_context_tokens, all_context_segments, all_context_masks, all_intent_labels,
              all_nextstep_labels, all_action_labels, all_value_labels, all_utterance_labels,
              all_action_masks, all_value_masks, all_utterance_masks)

  def examples_to_features(self, examples, label_lists, max_seq_length, tokenizer,
                output_mode, extra_sep_token=False, cls_token_segment_id=0,
                sequence_a_segment_id=1, pad_token_segment_id=0,
                process_count=cpu_count() - 2):
    # Loads a data file into a list of `InputBatch`s
    label_maps = [{label: i for i, label in enumerate(ll)} for ll in label_lists]
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if extra_sep_token else 2
    effective_max = max_seq_length - special_tokens_count

    self.special = {
      'tokens': [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token],
      'ids': [cls_token_segment_id, sequence_a_segment_id, pad_token_segment_id],
      'maximum': [effective_max, max_seq_length]
    }
    self.process_candidates(ready_to_convert=True)

    features = []
    for example in progress_bar(examples, total=len(examples)):
      grouped = (example, label_maps, output_mode)
      converted = self.convert_example(grouped)
      features.append(converted)

    return features

class CascadeDataset(CompletionDataset):

  def convert_example(self, example_row, pad_token=0):
    example, label_map, output_mode = example_row
    sep_token = self.special['tokens'][1]
    texts = [utterance.split('|')[1] for utterance in example.context]
    embedded, segments, mask = self.embed_utterance(f' {sep_token} '.join(texts))

    embedded_context = self.convert_context_tokens(example.context_tokens)
    label_ids = self.build_label_ids(example, label_map)
    return CascadingFeatures(input_ids=embedded, segment_ids=segments, input_mask=mask, 
      label_ids=label_ids, candidates=example.candidates, context=embedded_context,
      convo_id=example.convo_id, turn_count=example.turn_count)

  def collate_func(self, args, split, raw_data):
    examples = []
    for i, line in enumerate(raw_data[split]):
      examples.append(CascadingExample(f'{split}_{i}', *line))
    random.shuffle(examples)

    labels = [self.intent_labels, self.nextstep_labels, self.action_labels]
    features = self.examples_to_features(examples, labels, args.max_seq_len,
      self.tokenizer, 'completion', sequence_a_segment_id=0 if self.model_type in ['roberta', 'large'] else 1)
    
    all_input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_id for f in features], dtype=torch.long)
    all_mask_ids = torch.tensor([f.mask_id for f in features], dtype=torch.long)

    all_candidates = torch.tensor([f.candidates for f in features], dtype=torch.long)
    all_context_tokens = torch.tensor([f.context_token for f in features], dtype=torch.long)
    all_context_segments = torch.tensor([f.context_segment for f in features], dtype=torch.long)
    all_context_masks = torch.tensor([f.context_mask for f in features], dtype=torch.long)
    
    all_intent_labels = torch.tensor([f.intent_id for f in features], dtype=torch.long)
    all_nextstep_labels = torch.tensor([f.nextstep_id for f in features], dtype=torch.long)
    all_action_labels = torch.tensor([f.action_id for f in features], dtype=torch.long)
    all_value_labels = torch.tensor([f.value_id for f in features], dtype=torch.long)
    all_utterance_labels = torch.tensor([f.utt_id for f in features], dtype=torch.long)

    all_action_masks = torch.tensor([f.action_mask for f in features], dtype=torch.long)
    all_value_masks = torch.tensor([f.value_mask for f in features], dtype=torch.long)
    all_utterance_masks = torch.tensor([f.utt_mask for f in features], dtype=torch.long)

    all_convo_ids = torch.tensor([f.convo_id for f in features], dtype=torch.long)
    all_turn_counts = torch.tensor([f.turn_count for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_segment_ids, all_mask_ids, all_candidates,
              all_context_tokens, all_context_segments, all_context_masks, all_intent_labels,
              all_nextstep_labels, all_action_labels, all_value_labels, all_utterance_labels,
              all_action_masks, all_value_masks, all_utterance_masks, 
              all_convo_ids, all_turn_counts)

