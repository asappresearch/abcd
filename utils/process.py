import os, sys, pdb
import csv
import json
import random
import torch
import numpy as np
import pandas as pd
import datetime

from tqdm import tqdm as progress_bar
from components.datasets import ActionFeature, CompletionFeature, CascadeFeature
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def setup_dataloader(datasets, batch_size, split):
  dataset = datasets[split]
  num_examples = len(dataset)
  sampler = RandomSampler(dataset) if split == 'train' else SequentialSampler(dataset)
  collate = dataset.collate_func
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate)
  print(f"Loaded {split} data with {len(dataloader)} batches")
  return dataloader, num_examples

def notify_feature_sizes(args, features):
  if args.verbose:
    for split, feats in features.items():
      print(f"{split}: {len(feats)} features")

def prepare_action_labels(ontology):
  action_list = []
  for section, buttons in ontology["actions"].items():
    actions = buttons.keys()
    action_list.extend(actions)
  return {action: idx for idx, action in enumerate(action_list)}

def prepare_intent_labels(ontology):
  intent_list = []
  for flow, subflows in ontology["intents"]["subflows"].items():
    intent_list.extend(subflows)
  return {intent: idx for idx, intent in enumerate(intent_list)}

def prepare_nextstep_labels(ontology):
  nextstep_list = ontology['next_steps']
  return {nextstep: idx for idx, nextstep in enumerate(nextstep_list)}

def prepare_value_labels(ontology):
  value_list = []
  for category, values in ontology["values"]["enumerable"].items():
    # value_list.extend(values)
    for val in values:
      if val not in value_list:   # remove exactly one instance of credit_card
        value_list.append(val.lower())
  return {slotval: idx for idx, slotval in enumerate(value_list)}

class BaseProcessor(object):

  def __init__(self, args, tokenizer, ontology):
    self.task = args.task
    self.model_type = args.model_type
    self.use_intent = args.use_intent

    self.tokenizer = tokenizer
    self.ontology = ontology

    self.prepare_labels(args)
    self.prepare_special_tokens(args)

  def prepare_labels(self, args):
    self.non_enumerable = self.ontology["values"]["non_enumerable"]
    self.enumerable = {}
    for category, values in self.ontology["values"]["enumerable"].items():
      self.enumerable[category] = [val.lower() for val in values]

    self.mappers = {  
      'value': prepare_value_labels(self.ontology),
      'action': prepare_action_labels(self.ontology),
      'intent': prepare_intent_labels(self.ontology),
      'nextstep': prepare_nextstep_labels(self.ontology)
    }  # utterance is ranking, so not needed
    self.start_idx = len(self.mappers['value'])

    # Break down the slot values by action
    self.value_by_action = {}
    for section, actions in self.ontology["actions"].items():
      for action, targets in actions.items():
        self.value_by_action[action] = targets

  def prepare_special_tokens(self, args):
    special_tokens_count = 3 if args.model_type == 'roberta' else 2
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    effective_max = args.max_seq_len - special_tokens_count
    cls_token_segment_id = 0
    sequence_a_segment_id = 0 if args.model_type == 'roberta' else 1
    pad_token_segment_id = 0

    self.special = {
      'tokens': [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token],
      'ids': [cls_token_segment_id, sequence_a_segment_id, pad_token_segment_id],
      'maximum': [effective_max, args.max_seq_len]
    }

  def value_to_id(self, context, action, value, potential_vals):
    # context is a list of utterances
    target_id = -1
    action_tokens = self.tokenizer.tokenize(action)
    filtered = []
    for utterance in context:
      speaker, text = utterance.split('|')
      context_tokens = self.tokenizer.tokenize(text)
      for tok in context_tokens:
        if tok in filtered:  continue       # find uniques this way to preserve order
        if len(tok) > 2:
          filtered.append(tok)            # remove punctuation and special tokens

    effective_max = 100 - (len(action_tokens) + 3)   # three special tokens will be added
    tokens = filtered[-effective_max:]             # [CLS] action [SEP] filtered [SEP]

    for option in potential_vals:
      if option in self.enumerable:    # just look it up
        if value in self.enumerable[option]:
          target_id = self.mappers['value'][value]
      else:
        entity = f'<{option}>'       # calculate location in the context
        if entity in tokens:
          target_id = self.start_idx + tokens.index(entity)

      if target_id >= 0: break     # we found our guy, so let's move on

    return target_id, tokens

  def build_features(self, args, raw_data):
    print("Build features method missing")
    raise NotImplementedError()

  def embed_utterance(self, text):
    cls_token, sep_token, pad_token = self.special['tokens']
    cls_token_segment_id, sequence_a_segment_id, pad_token_segment_id = self.special['ids']
    effective_max, max_seq_length = self.special['maximum']

    text = pad_token if text == '' else text
    if self.model_type in ['roberta', 'large']:
      tokens = self.tokenizer.tokenize(text, add_prefix_space=True)
    else:
      tokens = self.tokenizer.tokenize(text)
    if len(tokens) > effective_max:
      tokens = tokens[:effective_max]

    tokens = tokens + [sep_token]
    segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * len(tokens)
    tokens = [cls_token] + tokens
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Since we only ever have input text, the "type_ids" are instead used to indicate
    # the speaker of 0 = customer, 1 = agent and 2 = action 
    # The embedding vectors for `type=0` and `type=1` were learned during pre-training and 
    # are added to the wordpiece embedding vector (and position vector). Hopefully
    # the fine-tuning can overcome this difference in semantic meaning

    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    pad_token_id = self.tokenizer.convert_tokens_to_ids([pad_token])[0]
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token_id] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
    input_mask = input_mask + ([0] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return input_ids, segment_ids, input_mask

  def convert_context_tokens(self, context_tokens):
    # context_tokens is a list of pre-tokenized strings, with action name in the front
    # and we want a list of embedded vectors
    cls_token, sep_token, pad_token = self.special['tokens']
    cls_token_segment_id, sequence_a_segment_id, pad_token_segment_id = self.special['ids']

    tokens = context_tokens + [sep_token]
    segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * len(tokens)
    tokens = [cls_token] + tokens

    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(token_ids)

    pad_token_id = self.tokenizer.convert_tokens_to_ids([pad_token])[0]
    # Zero-pad up to the sequence length.
    padding_length = 100 - len(token_ids)
    token_ids = token_ids + ([pad_token_id] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
    mask_ids = input_mask + ([0] * padding_length)

    return {'token_ids': token_ids, 'segment_ids': segment_ids, 'mask_ids': mask_ids}

  def action_to_id(self, action):
    if self.task == 'value':
      return action
    if ' ' in action:
      action, input_position = action.split(' ')
    return self.mappers['action'][action]

  def convert_example(self, dialog_history, target_ids, context_tokens, intent=None, candidates=None):
    sep_token = self.special['tokens'][1]

    texts = [utterance.split('|')[1] for utterance in dialog_history]  # drop the speaker
    if self.use_intent:
      texts = [f"{intent}|{text}" for text in texts]
    embedded, segments, mask = self.embed_utterance(f' {sep_token} '.join(texts))

    if self.task == 'ast':
      embedded_context = self.convert_context_tokens(context_tokens)
      feature = ActionFeature(input_ids=embedded, segment_ids=segments, input_mask=mask, 
            label_ids=target_ids, context=embedded_context)
    elif self.task == 'cds':
      embedded_context = self.convert_context_tokens(context_tokens)
      feature = CascadeFeature(input_ids=embedded, segment_ids=segments, input_mask=mask, 
            label_ids=target_ids, context=embedded_context, candidates=candidates)

    return feature
    
class ASTProcessor(BaseProcessor):

  def collect_one_example(self, context, action, value, potential_vals):
    # actions that don't require any values
    if value == 'not applicable':
      target_ids = { 'action': self.action_to_id(action), 'value': -1}
      feature = self.convert_example(context, target_ids, [])
      self.split_feats.append(feature)      

    else: # actions that require at least one value
      value_id, context_tokens = self.value_to_id(context, action, value, potential_vals)
      # context_tokens are used for copying from the context when selecting values
      if value_id >= 0:
        target_ids = { 'action': self.action_to_id(action), 'value': value_id}
        feature = self.convert_example(context, target_ids, context_tokens)
        self.split_feats.append(feature)

  def collect_examples(self, context, action, values):
    potential_vals = self.value_by_action[action]
    # just skip if action does not require value inputs
    if len(potential_vals) > 0:
      # these two actions require 3 value inputs, so we break it down
      if action in ['verify-identity', 'validate-purchase']:  
        # a smarter model can be made that handles each position conditioned on other values
        for position, value in zip(['a', 'b', 'c'], values):
          action_name = action + ' ' + position
          self.collect_one_example(context, action_name, value, potential_vals)
      # other actions require a single value to be filled
      else:
        self.collect_one_example(context, action, values[0], potential_vals)
    else:
      self.collect_one_example(context, action, 'not applicable', potential_vals)

  def build_features(self, args, raw_data):
    features = {}

    for split, data in raw_data.items():
      self.split_feats = []
      print(f"Building features for {split}")

      for convo in progress_bar(data, total=len(data)):
        so_far = []

        for turn in convo['delexed']:
          speaker, utt = turn['speaker'], turn['text']
          _, _, action, values, _ = turn['targets']

          if speaker in ['agent', 'customer']:
            utt_str = f'{speaker}|{utt}'
            so_far.append(utt_str)
          else:   # create a training example during every action
            context = so_far.copy() # [::-1] to reverse
            self.collect_examples(context, action, values)
            action_str = f'action|{action}'
            so_far.append(action_str)

      features[split] = self.split_feats
    return features

class CDSProcessor(BaseProcessor):

  def collect_one_example(self, dialog_history, targets, support_items):
    intent, nextstep, action, _, utt_id = targets
    candidates = [-1]*100
    context_tokens = []
    action_id, value_id = -1, -1

    if nextstep == 'take_action':
      value, potential_vals, convo_id, turn_id = support_items
      action_id = self.action_to_id(action)
      if value != 'not applicable':
        value_id, context_tokens = self.value_to_id(dialog_history, action, value, potential_vals)

    elif nextstep == 'retrieve_utterance':
      candidates, convo_id, turn_id = support_items

    elif nextstep == 'end_conversation':
      convo_id, turn_id = support_items

    target_ids = {
      'intent': self.mappers['intent'][intent],
      'nextstep': self.mappers['nextstep'][nextstep],
      'action': action_id,
      'value': value_id,
      'utterance': utt_id,
      'convo': convo_id,
      'turn': turn_id,
    }
    feature = self.convert_example(dialog_history, target_ids, context_tokens, intent, candidates)  
    self.split_feats.append(feature)

  def collect_examples(self, context, targets, convo_id, turn_id):
    _, _, action, values, _ = targets
    potential_vals = self.value_by_action[action]

    if len(potential_vals) > 0:           # just skip if action does not require inputs
      if action in ['verify-identity', 'validate-purchase']:  # 3 action inputs
        for position, value in zip(['a', 'b', 'c'], values):
          action_name = action + ' ' + position
          self.collect_one_example(context, targets, (value, potential_vals, convo_id, turn_id))  
      else:
        self.collect_one_example(context, targets, (values[0], potential_vals, convo_id, turn_id))  
    else:
      self.collect_one_example(context, targets, ("not applicable", potential_vals, convo_id, turn_id))

  def build_features(self, args, raw_data):
    features = {}

    for split, data in raw_data.items():
      self.split_feats = []
      print(f"Building features for {split}")

      for convo in progress_bar(data, total=len(data)):
        so_far = []

        for turn in convo['delexed']:
          speaker, text = turn['speaker'], turn['text']
          utterance = f"{speaker}|{text}"

          if speaker == 'agent':
            context = so_far.copy()
            support_items = turn['candidates'], convo['convo_id'], turn['turn_count']
            self.collect_one_example(context, turn['targets'], support_items)
            so_far.append(utterance)
          elif speaker == 'action':
            context = so_far.copy()
            self.collect_examples(context, turn['targets'], convo['convo_id'], turn['turn_count'])
            so_far.append(utterance)
          else:
            so_far.append(utterance)

        context = so_far.copy()  # the entire conversation
        end_targets = turn['targets'].copy()
        end_targets[1] = 'end_conversation'
        end_targets[4] = -1
        support_items = convo['convo_id'], turn['turn_count']
        self.collect_one_example(context, end_targets, support_items)

      features[split] = self.split_feats
    return features

def process_data(args, tokenizer, ontology, raw_data, cache_path, from_cache):
  # Takes in a pre-processed dataset and performs further operations:
  # 1) Extract the labels 2) Embed the inputs 3) Store both into features 4) Cache the results
  if args.task == 'ast':
    processor = ASTProcessor(args, tokenizer, ontology)
  elif args.task == 'cds':
    processor = CDSProcessor(args, tokenizer, ontology)

  if from_cache:
    features = torch.load(cache_path)
    print(f"Features loaded successfully.")
  else:
    features = processor.build_features(args, raw_data)
    print(f"Saving features into cached file {cache_path}")
    torch.save(features, cache_path)
  
  notify_feature_sizes(args, features)
  return features, processor.mappers
