import os, sys, pdb
import json
import math
import numpy as np
import GPUtil

import faiss
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, RobertaModel, AlbertModel
from transformers.file_utils import WEIGHTS_NAME

class CoreModel(nn.Module):
  def __init__(self, args, checkpoint_dir):
    super().__init__()
    if args.model_type == 'bert':
      self.encoder = BertModel.from_pretrained('bert-base-uncased')
    elif args.model_type == 'roberta':
      self.encoder = RobertaModel.from_pretrained('roberta-base')
    elif args.model_type == 'albert':
      self.encoder = AlbertModel.from_pretrained('albert-base-v2')

    self.outputs = ['intent', 'nextstep', 'action', 'value', 'utt']
    self.checkpoint_dir = checkpoint_dir

  def forward(self):
    raise NotImplementedError

  def save_pretrained(self):
    torch.save(self.state_dict(), self.checkpoint_dir)
    print(f"Model weights saved in {self.checkpoint_dir}")

  @classmethod
  def from_pretrained(cls, hidden_dim, ontology_size, base_model, device, checkpoint_dir):
    # Instantiate model.
    model = cls(hidden_dim, ontology_size, base_model, device)
    # Modify for cascading evaluation
    if 'cascade' in checkpoint_dir:
      checkpoint_dir = checkpoint_dir.replace('cascade_', '')
    # Load weights and fill them inside the model
    filepath = os.path.join(checkpoint_dir, 'pytorch_model.pt')
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model

class CascadeDialogSuccess(CoreModel):
  """ Unlike the BaseModel, will output 5 predictions, one for each component """
  def __init__(self, args, ontology_size):
    num_intents, num_nextsteps, num_actions, num_values, num_utterances = ontology_size
    super().__init__(args, num_intents)

    self.intent_projection = nn.Linear(args.hidden_dim, num_intents)
    self.nextstep_projection = nn.Linear(args.hidden_dim, num_nextsteps)
    self.action_projection = nn.Linear(args.hidden_dim, num_actions)

    self.candidate_linear = nn.Linear(args.hidden_dim, 128)
    self.context_linear = nn.Linear(args.hidden_dim, 128)

    self.enum_projection = nn.Linear(args.hidden_dim, num_values)
    self.copy_projection = nn.Linear(args.hidden_dim, 100)  # hardcode limit of 100 context tokens
    self.gating_mechanism = nn.Linear(args.hidden_dim * 2, 1)   # shrink down to scalar

    self.softmax = nn.Softmax(dim=1)
    self.sigmoid = nn.Sigmoid()

  def add_candidate_tooling(self, dataset):
    self.utt_vectors = dataset.utt_vectors
    self.utt_texts = dataset.utt_texts

  def forward(self, full_history, context_tokens, candidate_ids):
    _, pooled_history = self.encoder(**full_history)                # batch_size x 768
    intent_score = self.softmax(self.intent_projection(pooled_history))
    nextstep_score = self.softmax(self.nextstep_projection(pooled_history))
    action_score = self.softmax(self.action_projection(pooled_history))
    enum_prob = self.softmax(self.enum_projection(pooled_history))

    encoded_context = pooled_history.unsqueeze(1)               # (batch_size, 1, hidden_dim)
    projected_context = self.context_linear(encoded_context)  # (batch_size, 1, 128)

    candidates = []
    for row in candidate_ids:                             # each row includes 100 positions
      vectors = [self.utt_vectors[position] for position in row]
      candidates.append(vectors)                        # batch_size, num_candidates, hidden_dim
    candidates = torch.tensor(candidates, dtype=torch.float, device=self.device)
    candidates = self.candidate_linear(candidates)        # (batch_size, num_candidates, 128)
    candidates = candidates.transpose(1,2)                # (batch_size, 128, num_candidates)

    utt_score = torch.bmm(projected_context, candidates)
    utt_score = utt_score.squeeze(1)                 # (batch_size, num_candidates)
    utt_score = self.softmax(utt_score)             # changes scores to probabilities

    _, pooled_tokens = self.encoder(**context_tokens)               
    copy_prob = self.softmax(self.copy_projection(pooled_tokens))   # batch_size x 100
    reverse_copy_proj = self.copy_projection.weight.t()
    copy_context = torch.matmul(reverse_copy_proj, pooled_tokens)   # batch_size x hidden
    joined = torch.cat([pooled_history, copy_context], dim=1)       # batch_size x 768*2
    gate = self.sigmoid(self.gating_mechanism(joined))              # batch_size x 1

    enum_score = gate * enum_prob                                   # batch_size x 125
    copy_score = (1-gate) * copy_prob                               # batch_size x 100
    value_score = torch.cat([enum_score, copy_score], dim=1)        # batch_size x 225

    return intent_score, nextstep_score, action_score, value_score, utt_score

class TaskWithIntentModel(CoreModel):
  """ intents is a batch_size vector, we append this information to the encoded
  outputs to help with prediction of the various components since this model uses intents"""
  def __init__(self, hidden_dim, ontology_size, base_model, device):
    super().__init__(base_model, device)
    num_intents, num_nextsteps, num_actions, num_values, num_utterances = ontology_size

    self.candidate_linear = nn.Linear(hidden_dim, 128)
    self.copy_projection = nn.Linear(hidden_dim, 100)  # hardcode limit of 100 context tokens

    hidden_dim += 1    # to account for appending the intent label
    num_intents, num_nextsteps, num_actions, num_values, num_utterances = ontology_size
    self.num_intents = num_intents

    self.intent_projection = nn.Linear(hidden_dim, num_intents)   # REMOVE ME
    self.nextstep_projection = nn.Linear(hidden_dim, num_nextsteps)
    self.action_projection = nn.Linear(hidden_dim, num_actions)
    self.enum_projection = nn.Linear(hidden_dim, num_values)

    self.context_linear = nn.Linear(hidden_dim, 128)
    self.gating_mechanism = nn.Linear(hidden_dim + 100, 1)
    self.softmax = nn.Softmax(dim=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, full_history, context_tokens, candidate_ids, intents):
    _, pooled_history = self.encoder(**full_history)                # batch_size x 768

    intent_vector = intents.unsqueeze(1)                            # batch_size x 1
    intent_score = torch.zeros(intents.shape[0], self.num_intents).to(self.device)
    intent_score.scatter_(dim=1, index=intent_vector, value=1.0)    # batch_size x num_intents
    # batch_size x 769
    history_with_intent = torch.cat([pooled_history, intent_vector.type(torch.float)], dim=1)       
    nextstep_score = self.softmax(self.nextstep_projection(history_with_intent))
    action_score = self.softmax(self.action_projection(history_with_intent))
    enum_prob = self.softmax(self.enum_projection(history_with_intent))

    encoded_context = history_with_intent.unsqueeze(1)               # (batch_size, 1, hidden_dim)
    projected_context = self.context_linear(encoded_context)  # (batch_size, 1, 128)

    candidates = []
    for row in candidate_ids:                             # each row includes 100 positions
      vectors = [self.utt_vectors[position] for position in row]
      candidates.append(vectors)                        # batch_size, num_candidates, hidden_dim
    candidates = torch.tensor(candidates, dtype=torch.float, device=self.device)
    candidates = self.candidate_linear(candidates)        # (batch_size, num_candidates, 128)
    candidates = candidates.transpose(1,2)                # (batch_size, 128, num_candidates)

    utt_score = torch.bmm(projected_context, candidates)
    utt_score = utt_score.squeeze(1)                 # (batch_size, num_candidates)
    utt_score = self.softmax(utt_score)             # changes scores to probabilities

    _, pooled_tokens = self.encoder(**context_tokens)               
    copy_prob = self.softmax(self.copy_projection(pooled_tokens))   # batch_size x 100
    joined = torch.cat([history_with_intent, copy_prob], dim=1)     # batch_size x 869
    gate = self.sigmoid(self.gating_mechanism(joined))              # batch_size x 1

    enum_score = gate * enum_prob                                   # batch_size x 123
    copy_score = (1-gate) * copy_prob                               # batch_size x 100
    value_score = torch.cat([enum_score, copy_score], dim=1)        # batch_size x 223

    return intent_score, nextstep_score, action_score, value_score, utt_score

  def add_candidate_tooling(self, dataset):
    self.utt_vectors = dataset.utt_vectors
    self.utt_texts = dataset.utt_texts

class ActionStateTracking(CoreModel):
  """ An AST model should output predictions for buttons, slots and values.  There are multiple ways
  to accomplish this goal:
    a. Predicts all 3 parts separately and join the results afterwards
    b. Predict the 231 possible button-slots together and then just the values
    c. First predict the 30 available buttons alone and then the slot-values together
    d. First predict the 30 available buttons and then just the values, leaving the slots as implied
  Option D is reasonable because each value only belongs to a certain slot, so selecting the correct
  value implies that the slot has also been correctly selected.  This final option is implemented below.

  To perform value-filling, the task is further decomposed into copying unique tokens from the context
  for non-enumerable values (copy_score_ or selecting from the ontology for enumerable values (enum_score).
  """

  def __init__(self, args, mappers, checkpoint_dir):
    super().__init__(args, checkpoint_dir)
    self.outputs = ['action', 'value']
    self.mappings = mappers

    self.action_projection = nn.Linear(args.hidden_dim, len(mappers['action']))
    self.enum_projection = nn.Linear(args.hidden_dim, len(mappers['value']))
    self.copy_projection = nn.Linear(args.hidden_dim, 100)  # hardcode limit of 100 context tokens
    self.gating_mechanism = nn.Linear(args.hidden_dim + 100, 1)   # shrink down to scalar

    self.softmax = nn.Softmax(dim=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, full_history, context_tokens, masks):
    _, pooled_history = self.encoder(**full_history)                # batch_size x 768
    action_score = self.softmax(self.action_projection(pooled_history))
    enum_prob = self.softmax(self.enum_projection(pooled_history))

    _, pooled_tokens = self.encoder(**context_tokens)               
    copy_prob = self.softmax(self.copy_projection(pooled_tokens))   # batch_size x 100
    joined = torch.cat([pooled_history, copy_prob], dim=1)          # batch_size x 868
    gate = self.sigmoid(self.gating_mechanism(joined))              # batch_size x 1

    enum_score = gate * enum_prob                                   # batch_size x 125
    copy_score = (1-gate) * copy_prob                               # batch_size x 100
    value_score = torch.cat([enum_score, copy_score], dim=1)        # batch_size x 225

    return action_score, value_score

def get_loss(task):
  if task == 'utterance':
    """ 
    scores (predicted model output) - batch_size x num_candidates 
    targets (index of actual response) - batch_size
    """
    def rank_loss(scores, targets):
      # targets must be (batch_size x 1) to match the number of dimensions
      target_ids = targets.unsqueeze(1)        # batch_size, 1
      # scores is the input, dim=1 is the num_candidates dimension
      chosen = torch.gather(scores, dim=1, index=target_ids)
      correct = chosen.sum()                   # scalar

      shift = torch.max(scores)
      res = torch.exp(scores - shift)          # batch_size, num_candidates
      # perform log sum exp of the incorrect scores
      res = torch.log(torch.sum(res, dim=1))   # batch_size
      # add the shift back in to complete the log-sum-exp overflow trick
      incorrect = torch.sum(shift + res)       # scalar

      total = incorrect - correct
      return total, scores 

    return rank_loss

  elif task == 'aawv':
    loss_func = nn.CrossEntropyLoss()
    def action_with_inputs_loss(scores, targets, masks):
      action_score, value_score = scores
      action_target, value_target = targets

      action_loss = loss_func(action_score, action_target)
      value_loss = loss_func(value_score, value_target)

      total_loss = action_loss + value_loss
      return total_loss

    return action_with_inputs_loss

  elif task in ['tcom', 'tcwi', 'remove']:
    loss_func = get_loss('general')
    def task_completion_loss(scores, targets, masks):
      intent_score, nextstep_score, action_score, value_score, utt_score = scores
      intent_target, nextstep_target, action_target, value_target, utt_target = targets
  
      action_mask = masks['action'].type(torch.bool)
      action_target[~action_mask] = -100  # this is the default ignore index for nn.CrossEntropy()
      value_mask = masks['value'].type(torch.bool)
      value_target[~value_mask] = -100

      batch_size, num_candidates = utt_score.shape
      utt_score = utt_score * masks['utterance'].unsqueeze(1).repeat(1, num_candidates)
      utterance_target = utt_target * masks['utterance']

      intent_loss = loss_func(intent_score, intent_target)
      nextstep_loss = loss_func(nextstep_score, nextstep_target)
      action_loss = loss_func(action_score, action_target)
      value_loss = loss_func(value_score, value_target)
      
      utt_target_ids = utterance_target.unsqueeze(1)        # batch_size, 1
      chosen = torch.gather(utt_score, dim=1, index=utt_target_ids)
      correct = chosen.sum()                   # scalar

      shift = torch.max(utt_score)
      res = torch.exp(utt_score - shift)          # batch_size, num_candidates
      res = torch.log(torch.sum(res, dim=1))   # batch_size
      incorrect = torch.sum(shift + res)       # scalar
      utt_loss = incorrect - correct

      total_loss = intent_loss + nextstep_loss + action_loss + value_loss + utt_loss
      return total_loss
    return task_completion_loss
  else:  # this will be used by action value filling and ignored by the others
    return nn.CrossEntropyLoss()
