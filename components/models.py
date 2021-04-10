import os, sys, pdb
import json
import math
import numpy as np
import GPUtil

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
    self.use_intent = args.use_intent

  def forward(self):
    raise NotImplementedError

  def save_pretrained(self, filepath=None):
    if filepath is None:
      filepath = os.path.join(self.checkpoint_dir, 'pytorch_model.pt')
    torch.save(self.state_dict(), filepath)
    print(f"Model weights saved in {filepath}")

  @classmethod
  def from_pretrained(cls, hidden_dim, ontology_size, base_model, device, filepath=None):
    # Instantiate model.
    model = cls(hidden_dim, ontology_size, base_model, device)
    # Load weights and fill them inside the model
    if filepath is None:
      filepath = os.path.join(self.checkpoint_dir, 'pytorch_model.pt')
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model

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

  def forward(self, full_history, context_tokens):
    history_outputs = self.encoder(**full_history)                
    pooled_history = history_outputs.pooler_output                  # batch_size x 768
    action_score = self.softmax(self.action_projection(pooled_history))
    enum_prob = self.softmax(self.enum_projection(pooled_history))

    context_outputs = self.encoder(**context_tokens)               
    pooled_context = context_outputs.pooler_output                  # batch_size x hidden
    copy_prob = self.softmax(self.copy_projection(pooled_context))  # batch_size x 100
    reverse_copy_proj = self.copy_projection.weight.t()             # hidden x 100
    copy_context = torch.matmul(pooled_context, reverse_copy_proj)  # batch_size x 100
    joined = torch.cat([pooled_context, copy_context], dim=1)       # batch_size x 768+100
    gate = self.sigmoid(self.gating_mechanism(joined))              # batch_size x 1

    enum_score = gate * enum_prob                                   # batch_size x 126
    copy_score = (1-gate) * copy_prob                               # batch_size x 100
    value_score = torch.cat([enum_score, copy_score], dim=1)        # batch_size x 226

    return action_score, value_score

class CascadeDialogSuccess(CoreModel):
  """ Unlike the BaseModel, will output 5 predictions, one for each component """
  def __init__(self, args, mappers, checkpoint_dir):
    super().__init__(args, checkpoint_dir)
    self.outputs = ['intent', 'nextstep', 'action', 'value', 'utterance']
    self.mappings = mappers

    self.intent_projection = nn.Linear(args.hidden_dim, len(mappers['intent']))
    self.nextstep_projection = nn.Linear(args.hidden_dim, len(mappers['nextstep']))
    self.action_projection = nn.Linear(args.hidden_dim, len(mappers['action']))

    self.candidate_linear = nn.Linear(args.hidden_dim, 128)
    self.context_linear = nn.Linear(args.hidden_dim, 128)

    self.enum_projection = nn.Linear(args.hidden_dim, len(mappers['value']))
    self.copy_projection = nn.Linear(args.hidden_dim, 100)  # hardcode limit of 100 context tokens
    self.gating_mechanism = nn.Linear(args.hidden_dim + 100, 1)   # shrink down to scalar

    self.softmax = nn.Softmax(dim=1)
    self.sigmoid = nn.Sigmoid()

  def add_candidate_data(self, utt_texts, utt_vectors):
    self.utt_texts = utt_texts
    self.utt_vectors = utt_vectors

  def forward(self, full_history, context_tokens, tools):
    if self.use_intent:
      all_candidates, device, _ = tools
    else:
      all_candidates, device = tools

    history_outputs = self.encoder(**full_history)                # batch_size x 768
    pooled_history = history_outputs.pooler_output
    intent_score = self.softmax(self.intent_projection(pooled_history))
    nextstep_score = self.softmax(self.nextstep_projection(pooled_history))
    action_score = self.softmax(self.action_projection(pooled_history))
    enum_prob = self.softmax(self.enum_projection(pooled_history))

    encoded_history = pooled_history.unsqueeze(1)               # (batch_size, 1, hidden_dim)
    projected_history = self.context_linear(encoded_history)    # (batch_size, 1, 128)

    batch_cands = []
    for row in all_candidates:                            # each row includes 100 positions
      vectors = [self.utt_vectors[position] for position in row]
      batch_cands.append(torch.stack(vectors))             

    candidates = torch.stack(batch_cands).to(device)      # batch_size, num_candidates, hidden_dim
    candidates = self.candidate_linear(candidates)        # (batch_size, num_candidates, 128)
    candidates = candidates.transpose(1,2)                # (batch_size, 128, num_candidates)

    utt_score = torch.bmm(projected_history, candidates)
    utt_score = utt_score.squeeze(1)                      # (batch_size, num_candidates)
    utt_score = self.softmax(utt_score)                   # normalize into probabilities

    context_outputs = self.encoder(**context_tokens)               
    pooled_context = context_outputs.pooler_output
    copy_prob = self.softmax(self.copy_projection(pooled_context))  # batch_size x 100
    reverse_copy_proj = self.copy_projection.weight.t()
    copy_context = torch.matmul(pooled_context, reverse_copy_proj)  # batch_size x hidden
    joined = torch.cat([pooled_context, copy_context], dim=1)       # batch_size x 768+100
    gate = self.sigmoid(self.gating_mechanism(joined))              # batch_size x 1

    enum_score = gate * enum_prob                                   # batch_size x 125
    copy_score = (1-gate) * copy_prob                               # batch_size x 100
    value_score = torch.cat([enum_score, copy_score], dim=1)        # batch_size x 225

    return intent_score, nextstep_score, action_score, value_score, utt_score
