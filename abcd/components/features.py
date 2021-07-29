
class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, input_context, target_label, candidates=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      context: list of strings. The untokenized text of the converation so far.
      label: (Optional) string. The label of the example. This should be
      specified for train and dev examples, but not for test examples.
      candidates: list of candidates to choose from for utterance ranking
    """
    self.guid = guid
    self.context = input_context
    self.label = target_label
    self.candidates = candidates

class ActionExample(InputExample):
  """A single training/test example for slot value filling. """
  def __init__(self, guid, input_context, target_label, tokens, action):
    super().__init__(guid, input_context, target_label)
    self.context_tokens = tokens
    self.action = action

class CompleteExample(InputExample):
  """A single training/test example for task completion. """
  def __init__(self, guid, input_context, targets, tokens, candidates):
    super().__init__(guid, input_context, None, candidates)
    self.context_tokens = tokens

    intent, nextstep, action, value_index, utt_index = targets
    self.intent_label = intent
    self.nextstep_label = nextstep
    self.action_label = action
    self.value_label = value_index
    self.utt_label = utt_index

class CascadingExample(InputExample):
  """A single training/test example for task completion. """
  def __init__(self, guid, input_context, targets, tokens, candidates, convo_id, turn_count):
    super().__init__(guid, input_context, None, candidates)
    self.context_tokens = tokens

    intent, nextstep, action, value_index, utt_index = targets
    self.intent_label = intent
    self.nextstep_label = nextstep
    self.action_label = action
    self.value_label = value_index
    self.utt_label = utt_index

    self.convo_id = convo_id
    self.turn_count = turn_count

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, input_ids, segment_ids, input_mask, label_id, position_ids=None):
    self.input_id = input_ids
    self.segment_id = segment_ids
    self.mask_id = input_mask
    self.label_id = label_id
    self.position_id = position_ids

class CandidateFeatures(InputFeatures):
  """ A single set of features with precomputed candidates """
  def __init__(self, input_ids, segment_ids, input_mask, label_id, candidates, position_ids=None):
    super().__init__(input_ids, segment_ids, input_mask, label_id, position_ids)
    # candidates is a (batch_size x num_candidates x hidden_dim) tensor
    self.candidates = candidates

class ActionFeatures(InputFeatures):
  """ A single set of features with precomputed context token ids"""
  def __init__(self, input_ids, segment_ids, input_mask, label_id, 
        context, action_id, position_ids=None):
    super().__init__(input_ids, segment_ids, input_mask, label_id, position_ids)
    # token_ids is a batch_size length list, where each item is 100 ids
    self.context_token = context['token_ids']
    self.context_segment = context['segment_ids']
    self.context_mask = context['mask_ids']
    self.action_id = action_id

class CompletionFeatures(InputFeatures):
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

class CascadingFeatures(InputFeatures):
  """ A single set of completion features with precomputed context token ids"""
  def __init__(self, input_ids, segment_ids, input_mask, label_ids, 
          candidates, context, convo_id, turn_count):
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

    self.convo_id = convo_id
    self.turn_count = turn_count
