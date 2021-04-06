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


def get_intent_label(scene):
  if scene['flow'] in ["storewide_query", "single_item_query"]:
    intent = scene['subflow'].split('_')[0]
  elif scene['subflow'] == 'status_questions':
    intent = 'status_active'
  elif scene['subflow'] == 'status_delivery_date':
    intent = 'status_delivery_time'
  else:
    intent = scene['subflow']
  return intent

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

def preprocess_TCWI(self, filename):
  scenario_df = self.load(f"scenarios_{filename}", 'csv')
  utterance_df = self.load(f"utterances_{filename}", 'csv')
  contexts, size = self.group_by_conversation(utterance_df, use_turn=True, use_value=True)
  
  num_samples = 40000
  cand_texts, cand_ids, id_mapping = UtteranceLoader.sample_candidates(contexts, size, num_samples)

  self.data = []
  for convo_id, conversation in progress_bar(contexts.items(), total=size):
    scene = self.extract_scene(scenario_df, convo_id)
    delexed = self.delexicalization(scene, conversation, use_turn=True)

    so_far = []
    for turn in delexed:
      speaker, text, action, turn_count, values = turn
      # Get the intent label
      intent = get_intent_label(scene)
      utt = intent + '|' + text
        
      if speaker == 'customer':
        so_far.append(utt)
      elif speaker == 'agent':
        pos = set(text.split())  # get all unique tokens
        candidates = []
        while len(candidates) < 99:
          ni = random.choice( range(0,num_samples) )
          neg_id, negative = cand_ids[ni], cand_texts[ni]
          neg = set(negative.split())
          if ni not in candidates and jaccard_distance(pos, neg) > 0.2:
            candidates.append(ni)

        pos_id = str(convo_id) + '_' + str(turn_count)
        if pos_id in cand_ids:
          position = id_mapping[pos_id]
        else:
          position = len(cand_ids)
          cand_texts.append(text)
          cand_ids.append(pos_id)
          id_mapping[pos_id] = position

        utt_index = random.choice(range(0,100))
        candidates.insert(utt_index, position)

        context = so_far.copy()
        #          intent, nextstep, action, value_index, utt_index
        targets = [intent, 'retrieve_utterance', None, None, utt_index]
        self.data.append((context, targets, [], candidates))
        so_far.append(utt)
      else:  # system action
        nextstep = 'take_action'
        context = so_far.copy() # [::-1] to reverse
        self.collect_examples(context, intent, nextstep, action, values)
        so_far.append(f'action|{action}')

    context = so_far.copy()
    targets = [intent, 'end_conversation', None, None, None]
    self.data.append((context, targets, [], [-1]*100))

  trainset, devset, testset = self.split_dataset(self.data)
  self.dataset = {"train": trainset, "dev": devset, "test": testset}
  self.dataset["all_utterances"] = cand_texts

class BaseProcessor(object):

  def __init__(self, args, tokenizer, ontology):
    self.task = args.task
    self.model_type = args.model_type
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
    sequence_a_segment_id = 1
    pad_token_segment_id = 0

    self.special = {
      'tokens': [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token],
      'ids': [cls_token_segment_id, sequence_a_segment_id, pad_token_segment_id],
      'maximum': [effective_max, args.max_seq_len]
    }

  def extract_target(self, context, action, value, options):
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

    for option in options:
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

  def act2id(self, action):
    if self.task == 'value':
      return action
    if ' ' in action:
      action, input_position = action.split(' ')
    return self.mappers['action'][action]

  def build_label_ids(self, example, label_maps):
    intent_map, nextstep_map, action_map = label_maps
    intent, nextstep = example.intent_label, example.nextstep_label
    label_ids = {'intent': intent_map[intent], 'nextstep': nextstep_map[nextstep] }
    num_intents, num_nextsteps, num_actions, num_values, num_utterances = self.num_labels
    
    assert(len(example.candidates) == 100)

    # Retrieve Utterance
    if nextstep == 'retrieve_utterance':
      assert(example.action_label is None)
      assert(example.value_label is None)
      assert(len(example.context_tokens) == 0)
      label_ids['action'] = -1  # will be out of bounds
      label_ids['value'] = -1 # will be out of bounds
      label_ids['utterance'] = example.utt_label

    elif nextstep == 'take_action':
      assert(example.utt_label is None)
      label_ids['action'] = self.convert_action_to_id(action_map, example.action_label)
      label_ids['value'] = example.value_label
      label_ids['utterance'] = -1  # will be out of bounds

    elif nextstep == 'end_conversation':
      assert(example.action_label is None)
      assert(example.value_label is None)
      assert(example.utt_label is None)
      label_ids['action'] = -1  # will be out of bounds
      label_ids['value'] = -1  # will be out of bounds
      label_ids['utterance'] = -1  # will be out of bounds

    else:
      raise KeyError("nextstep key is incorrect, not part of valid set")

    return label_ids

  def convert_example(self, dialog_history, context_tokens, target_id, action):
    sep_token = self.special['tokens'][1]
    texts = [utterance.split('|')[1] for utterance in dialog_history]  # drop the speaker
    embedded, segments, mask = self.embed_utterance(f' {sep_token} '.join(texts))
    # embedded, segments, mask = self.embed_utterance(f' {sep_token} '.join(dialog_history))

    if self.task == 'ast':
      embedded_context = self.convert_context_tokens(context_tokens)
      feature = ActionFeature(input_ids=embedded, segment_ids=segments, input_mask=mask, 
            label_id=target_id, context=embedded_context, action_id=self.act2id(action))
    elif self.task == 'cds':
      embedded_context = self.convert_context_tokens(context_tokens)
      target_ids = self.build_label_ids(example, label_map)
      feature = CascadeFeature(input_ids=embedded, segment_ids=segments, input_mask=mask, 
            label_ids=target_ids, context=embedded_context, candidates=example.candidates)

    return feature
    
class ASTProcessor(BaseProcessor):

  def collect_one_example(self, context, action, value, candidates):
    target_id, tokens = self.extract_target(context, action, value, candidates)
    # tokens are used for copying from the context when selecting values
    if target_id >= 0:
      feature = self.convert_example(context, tokens, target_id, action)
      self.split_feats.append(feature)

  def collect_examples(self, context, action, values):
    candidates = self.value_by_action[action]
    # just skip if action does not require value inputs
    if len(candidates) > 0:

      # these two actions require 3 value inputs, so we break it down
      if action in ['verify-identity', 'validate-purchase']:  
        # a smarter model can be made that handles each position conditioned on other values
        for position, value in zip(['a', 'b', 'c'], values):
          action_name = action + ' ' + position
          self.collect_one_example(context, action_name, value, candidates)
      # other actions require a single value to be filled
      else:
        self.collect_one_example(context, action, values[0], candidates)

  def build_features(self, args, raw_data):
    features = {}

    for split, data in raw_data.items():
      self.split_feats = []
      print(f"Building features for {split}")

      for convo in progress_bar(data, total=len(data)):
        so_far = []

        for turn in convo['delex']:
          speaker, utt, action, value = turn

          if speaker in ['agent', 'customer']:
            utt_str = f'{speaker}|{utt}'
            so_far.append(utt_str)
          else:   # create a training example during every action
            context = so_far.copy() # [::-1] to reverse
            self.collect_examples(context, action, value)
            action_str = f'action|{action}'
            so_far.append(action_str)

      features[split] = self.split_feats
    return features

def CDSProcessor(BaseProcessor):

  def collect_examples(self, context, intent, nextstep, action, values, convo_id, turn_count):
    options = self.value_by_action[action]
    candidates = [-1] * 100

    if len(options) > 0:           # just skip if action does not require inputs
      if action in ['verify-identity', 'validate-purchase']:  # 3 action inputs
        for position, value in zip(['a', 'b', 'c'], values):
          action_name = action + ' ' + position
          value_index, tokens = self.extract_target(context, action_name, value, options)
          targets = [intent, nextstep, action_name, value_index, None]
          self.data.append((context, targets, tokens, candidates, convo_id, turn_count))
      else:
        value_index, tokens = self.extract_target(context, action, values[0], options)
        targets = [intent, nextstep, action, value_index, None]
        self.data.append((context, targets, tokens, candidates, convo_id, turn_count))
    else:
      targets = [intent, nextstep, action, -1, None]
      self.data.append((context, targets, [], candidates, convo_id, turn_count))

  def build_features(self, args, raw_data):
    scenario_df = self.load(f"scenarios_{filename}", 'csv')
    utterance_df = self.load(f"utterances_{filename}", 'csv')
    contexts, size = self.group_by_conversation(utterance_df, use_turn=True, use_value=True)
    
    num_samples = 40000
    cand_texts, cand_ids, id_mapping = UtteranceLoader.sample_candidates(contexts, size, num_samples)

    self.data = []
    for convo_id, conversation in progress_bar(contexts.items(), total=size):
      scene = self.extract_scene(scenario_df, convo_id)
      delexed = self.delexicalization(scene, conversation, use_turn=True)

      so_far = []
      for turn in delexed:
        speaker, text, action, turn_count, values = turn
        intent = self.get_intent_label(scene)
        utt = speaker + '|' + text
          
        if speaker == 'customer':
          so_far.append(utt)
        elif speaker == 'agent':
          pos = set(text.split())  # get all unique tokens
          candidates = []
          while len(candidates) < 99:
            ni = random.choice( range(0,num_samples) )
            neg_id, negative = cand_ids[ni], cand_texts[ni]
            neg = set(negative.split())
            if ni not in candidates and jaccard_distance(pos, neg) > 0.2:
              candidates.append(ni)

          pos_id = str(convo_id) + '_' + str(turn_count)
          if pos_id in cand_ids:
            position = id_mapping[pos_id]
          else:
            position = len(cand_ids)
            cand_texts.append(text)
            cand_ids.append(pos_id)
            id_mapping[pos_id] = position

          utt_index = random.choice(range(0,100))
          candidates.insert(utt_index, position)

          context = so_far.copy()
          #          intent, nextstep, action, value_index, utt_index
          targets = [intent, 'retrieve_utterance', None, None, utt_index]
          self.data.append((context, targets, [], candidates, convo_id, turn_count))
          so_far.append(utt)
        else:  # system action
          nextstep = 'take_action'
          context = so_far.copy() # [::-1] to reverse
          self.collect_cds_examples(context, intent, nextstep, action, values, convo_id, turn_count)
          so_far.append(f'action|{action}')

      context = so_far.copy()
      targets = [intent, 'end_conversation', None, None, None]
      self.data.append((context, targets, [], [-1]*100, convo_id, turn_count))

    trainset, devset, testset = self.split_dataset(self.data)
    self.dataset = {"train": trainset, "dev": devset, "test": testset}
    self.dataset["all_utterances"] = cand_texts

def process_data(args, tokenizer, ontology, raw_data, cache_path, from_cache):
  # Takes in a pre-processed dataset and performs further operations:
  # 1) Extract the labels 2) Embed the inputs 3) Store both into features 4) Cache the results
  if args.task == 'ast':
    processor = ASTProcessor(args, tokenizer, ontology)
  elif args.task == 'cds':
    processor = CDSProcessor(args, tokenizer, ontology)

  if from_cache:
    features = torch.load(cache_path)
  else:
    features = processor.build_features(args, raw_data)
    print(f"Saving features into cached file {cache_path}")
    torch.save(features, cache_path)
  
  notify_feature_sizes(args, features)
  return features, processor.mappers
