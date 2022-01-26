import os, sys, pdb
import random
import json
import torch
import numpy as np
import time as tm
import pandas as pd

from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict, Counter
from sklearn.metrics import accuracy_score

from components.systems import Application
from utils.help import prepare_inputs
from utils.load import load_guidelines

def ast_report(predictions, labels):
  bslot_preds, value_preds = predictions
  bslot_labels, value_labels = labels

  size = len(bslot_preds)
  assert(size == len(value_labels))

  top_bslot_preds = np.argmax(bslot_preds, axis=1)
  bslot_match = bslot_labels == top_bslot_preds   # array of booleans
  bslot_acc = sum(bslot_match) / float(size) 

  top_value_preds = np.argmax(value_preds, axis=1)
  value_match = value_labels == top_value_preds
  value_acc = sum(value_match) / float(size) 

  joint_match = bslot_match & value_match
  joint_acc = sum(joint_match) / float(size) 

  full_result = {'Bslot_Accuracy': round(bslot_acc, 4),
          'Value_Accuracy': round(value_acc, 4),
          'Joint_Accuracy': round(joint_acc, 4),}

  return full_result, 'Joint_Accuracy'

def ranking_report(predictions, labels, use_match=False):
  full_result = {}
  utt_match = []

  for rank in [1,5,10]:
    level = -rank   # select the top 5 rather than bottom 5
    num_correct, num_possible = 0, 0
    # vectorized version possible, but a lot less readable
    for pred, label in zip(predictions, labels):
      top_k_indexes = np.argpartition(pred, kth=level)[level:]
      if label in top_k_indexes:
        num_correct += 1
        if rank == 1:
          utt_match.append(True)
      else:
        if rank == 1:
          utt_match.append(False)

      if label >= 0:    # -1 means the turn was take-action or end-of-convo
        num_possible += 1

    rank_name = f'Recall_at_{rank}'
    full_result[rank_name] = num_correct / num_possible

  if use_match:
    return full_result, utt_match
  else:
    return full_result, 'Recall_at_5'

def cds_report(predictions, labels, ci_and_tc, kb_labels=None):
  """ Calculated in the form of cascaded evaluation
  where each agent example or utterance a scored example"""
  intent_pred, nextstep_pred, bslot_pred, value_pred, utterance_rank = predictions
  intent_label, nextstep_label, bslot_label, value_label, utterance_label = labels
  convo_ids = ci_and_tc[0].detach().cpu().numpy()
  turn_counts = ci_and_tc[1].detach().cpu().numpy()

  if kb_labels is None:
    use_kb = False
  else:
    use_kb = True
    intent_list = kb_labels['intent']
    action_list = kb_labels['action']
    guidelines = load_guidelines()
    action_mask_map, intent_mask_map = Application.prepare_masks(*guidelines)

  num_turns = len(nextstep_pred)
  assert(num_turns == len(convo_ids))

  top_intent_preds = np.argmax(intent_pred, axis=1)
  intent_match = intent_label == top_intent_preds   # array of booleans
  intent_acc = sum(intent_match) / float(num_turns) 

  top_nextstep_preds = np.argmax(nextstep_pred, axis=1)
  nextstep_match = nextstep_label == top_nextstep_preds   # array of booleans
  nextstep_acc = sum(nextstep_match) / float(num_turns) 

  if use_kb:
    intent_masks = []
    for top_intent in top_intent_preds:
      intent_name = intent_list[top_intent]
      # each intent mask should be size of 30 long
      intent_mask = intent_mask_map[intent_name]
      intent_masks.append(intent_mask)
    # now, all non valid actions should go to zero
    bslot_pred *= np.array(intent_masks)

  top_bslot_preds = np.argmax(bslot_pred, axis=1)
  bslot_match = bslot_label == top_bslot_preds   # array of booleans
  num_turns_include_action = sum(bslot_label >= 0)
  bslot_acc = sum(bslot_match) / float(num_turns_include_action) 

  if use_kb:
    action_masks = []
    for top_action in top_bslot_preds:
      action_name = action_list[top_action]
      # each action mask should be size of 223 long
      action_mask = action_mask_map[action_name]
      action_masks.append(action_mask)
    # now, all non valid values should go to zero
    value_pred *= np.array(action_masks)

  top_value_preds = np.argmax(value_pred, axis=1)
  value_match = value_label == top_value_preds
  num_turns_include_value = sum(value_label >= 0)
  value_acc = sum(value_match) / float(num_turns_include_value) 

  joint_match = bslot_match & value_match
  joint_acc = sum(joint_match) / float(num_turns_include_action) 

  recall, utt_match = {}, []
  for rank in [1,5,10]:
    level = -rank   # select the top 5 rather than bottom 5
    num_correct, num_possible = 0, 0
    for pred, label in zip(utterance_rank, utterance_label):
      top_k_indexes = np.argpartition(pred, kth=level)[level:]
      if label in top_k_indexes:
        num_correct += 1
        if rank == 1:
          utt_match.append(True)
      else:
        if rank == 1:
          utt_match.append(False)

      if label >= 0:
        num_possible += 1
    recall[str(rank)] = num_correct / num_possible 

  # group by convo_ids
  unique_convo_ids = list(set(convo_ids))
  conversations = {}
  for uci in unique_convo_ids:
    turns, correctness = [], []
    row_id = 0
    for convo_id, turn_count in zip(convo_ids, turn_counts):
      if convo_id == uci:
        turns.append(turn_count)

        correct = False
        intent_right = intent_match[row_id]
        nextstep_right = nextstep_match[row_id]

        if nextstep_label[row_id] == 0:
          if intent_right and nextstep_right and utt_match[row_id]:
            correct = True
        elif nextstep_label[row_id] == 1:
          if intent_right and nextstep_right and joint_match[row_id]:
            correct = True
        elif nextstep_label[row_id] == 2:
          if intent_right and nextstep_right:
            correct = True

        correctness.append(correct)
      row_id += 1

    # sort by turn_counts
    ordered = [cor for _, cor in sorted( zip(turns,correctness), key=lambda tc: tc[0] )]
    conversations[uci] = ordered

  # count how many correct
  turn_score, turn_correct = 0, 0
  for convo_id, convo_correctness in conversations.items():
    convo_length = len(convo_correctness)
    # we use turn_id rather than the true turn_count since turn counts will skip numbers
    # when looping through the conversation due to skipping over customer utterances
    for turn_id in range(convo_length):
      num_remaining = convo_length - turn_id
      
      num_correct = 0
      # count up how many were predicted correctly
      while turn_id < convo_length and convo_correctness[turn_id]:
        num_correct += 1
        turn_id += 1

      if num_correct > 0:
        turn_correct += 1
      # normalize by the number of turns remaining
      turn_score += num_correct / num_remaining

  # normalize by total number of turns possible
  turn_acc = turn_correct / float(num_turns)
  final_score = turn_score / float(num_turns)

  full_result = {'Intent_Accuracy': round(intent_acc, 4),
         'Nextstep_Accuracy': round(nextstep_acc, 4),
           'Action_Accuracy': round(bslot_acc, 4),
          'Value_Accuracy': round(value_acc, 4),
          'Joint_Accuracy': round(joint_acc, 4),
             'Recall_at_1': round(recall['1'], 4),
             'Recall_at_5': round(recall['5'], 4),
            'Recall_at_10': round(recall['10'], 4),
           'Turn_Accuracy': round(turn_acc, 4),
           'Cascading_Score': round(final_score, 4) }

  return full_result, 'Cascading_Score'

def task_completion_report(predictions, labels, kb_labels=None):
  intent_pred, nextstep_pred, bslot_pred, value_pred, utterance_rank = predictions
  intent_label, nextstep_label, bslot_label, value_label, utterance_label = labels
  num_turns = len(nextstep_pred)

  if kb_labels is None:
    use_kb = False
  else:
    use_kb = True
    intent_list = kb_labels['intent']
    action_list = kb_labels['action']
    guidelines = load_guidelines()
    action_mask_map, intent_mask_map = Application.prepare_masks(*guidelines)

  top_intent_preds = np.argmax(intent_pred, axis=1)
  intent_match = intent_label == top_intent_preds   # array of booleans
  intent_acc = sum(intent_match) / float(num_turns) 

  top_nextstep_preds = np.argmax(nextstep_pred, axis=1)
  nextstep_match = nextstep_label == top_nextstep_preds   # array of booleans
  nextstep_acc = sum(nextstep_match) / float(num_turns) 

  if use_kb:
    intent_masks = []
    for top_intent in top_intent_preds:
      intent_name = intent_list[top_intent]
      # each intent mask should be size of 30 long
      intent_mask = intent_mask_map[intent_name]
      intent_masks.append(intent_mask)
    # now, all non valid actions should go to zero
    bslot_pred *= np.array(intent_masks)

  top_bslot_preds = np.argmax(bslot_pred, axis=1)
  bslot_match = bslot_label == top_bslot_preds   # array of booleans
  num_turns_include_action = sum(bslot_label >= 0) 
  bslot_acc = sum(bslot_match) / float(num_turns_include_action) 

  if use_kb:
    action_masks = []
    for top_action in top_bslot_preds:
      action_name = action_list[top_action]
      # each action mask should be size of 223 long
      action_mask = action_mask_map[action_name]
      action_masks.append(action_mask)
    # now, all non valid values should go to zero
    value_pred *= np.array(action_masks)

  top_value_preds = np.argmax(value_pred, axis=1)
  value_match = value_label == top_value_preds
  num_turns_include_value = sum(value_label >= 0)
  value_acc = sum(value_match) / float(num_turns_include_value) 

  joint_match = bslot_match & value_match
  joint_acc = sum(joint_match) / float(num_turns_include_action) 

  recall, utt_match = ranking_report(utterance_rank, utterance_label, use_match=True)

  assert(num_turns == len(value_label))
  assert(len(intent_pred) == len(nextstep_label))
  assert(len(utt_match) == num_turns)    
  assert(len(bslot_match) == len(top_value_preds))  

  turn_correct = 0
  for turn in range(num_turns):
    if intent_match[turn] and nextstep_match[turn]:
      pass
    else:
      continue

    if nextstep_label[turn] == 0 and utt_match[turn]:
      turn_correct += 1
    elif nextstep_label[turn] == 1 and joint_match[turn]:
      turn_correct += 1
    elif nextstep_label[turn] == 2:      # end_conversation
      turn_correct += 1
  turn_acc = turn_correct / float(num_turns)

  full_result = {'Intent_Accuracy': round(intent_acc, 4),
         'Nextstep_Accuracy': round(nextstep_acc, 4),
           'Action_Accuracy': round(bslot_acc, 4),
          'Value_Accuracy': round(value_acc, 4),
          'Joint_Accuracy': round(joint_acc, 4),
             'Recall_at_1': round(recall['Recall_at_1'], 4),
             'Recall_at_5': round(recall['Recall_at_5'], 4),
            'Recall_at_10': round(recall['Recall_at_10'], 4),
           'Turn_Accuracy': round(turn_acc, 4) }

  return full_result, 'Turn_Accuracy' 

def qualify(args, ids, tokenizer, target_maps, scores, targets):
  history_ids, context_ids = ids
  bslot_mapper, value_mapper = target_maps
  num_values = len(value_mapper)
  pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

  bslot_score, value_score = scores
  bslot_target, value_target = targets
  top_bslot_ids = np.argmax(bslot_score.detach().cpu().numpy(), axis=1)
  top_value_ids = np.argmax(value_score.detach().cpu().numpy(), axis=1)
   
  for index, (history, context) in enumerate(zip(history_ids, context_ids)):
    stripped_history = [x for x in history if x != pad_id]
    history_tokens = tokenizer.convert_ids_to_tokens(stripped_history)
    history_symbols = ' '.join(history_tokens).replace(' ##', '')
    history_text = history_symbols.replace('Ġ', '').replace('</s>', '//').replace(' âĢ Ļ', '\'')
    bslot_pred = bslot_mapper[top_bslot_ids[index]]
    bslot_actual = bslot_mapper[bslot_target[index].cpu()]
    
    if args.filter and (bslot_pred == bslot_actual):
      print('--- Skipping since model is correct ---')
      continue

    context_tokens = tokenizer.convert_ids_to_tokens(context)
    tvii = top_value_ids[index]
    if tvii >= num_values:
      tvii -= num_values
      value_pred = context_tokens[tvii]
    else:
      value_pred = value_mapper[tvii]

    vtic = value_target[index].cpu()  
    if vtic >= num_values:
      vtic -= num_values
      value_actual = context_tokens[vtic]
    else:
      value_actual = value_mapper[vtic]
    print(index, history_text)
    print(f"Predicted Button-slot: {bslot_pred}, Actual: {bslot_actual}")
    print(f"Predicted Value: {value_pred}, Actual: {value_actual}")

  pdb.set_trace()  

def quantify(args, predictions, labels, utils=None):
  assert len(predictions) == len(labels)
 
  if utils == "train" and not args.verbose:
    return predictions, labels

  if args.task == 'ast':
    predictions = [pred.detach().cpu().numpy() for pred in predictions]
    labels = [label.detach().cpu().numpy() for label in labels]
    report, res_name = ast_report(predictions, labels)

  elif args.task == 'cds':
    predictions = [pred.detach().cpu().numpy() for pred in predictions]
    labels = [label.detach().cpu().numpy() for label in labels]
    kb_labels = utils['kb_labels'] if args.use_kb else None

    if args.cascade:
      ci_and_tc = utils['ci_and_tc']
      result = cds_report(predictions, labels, ci_and_tc, kb_labels)
      report, res_name = result
    else:
      report, res_name = task_completion_report(predictions, labels, kb_labels)

  return report, res_name


if __name__ == '__main__':
  class MyModel():
    def __init__(self):
      self.utt_vectors = []
      self.utt_texts = []

  args = {}
  run_interaction(args, MyModel())
