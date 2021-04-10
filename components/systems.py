import os, sys, pdb
import random
import numpy as np
import json
import pandas as pd

class Application(object):
  def __init__(self, args, model, processor):
    self.task = args.task
    self.utt_vectors = model.utt_vectors
    self.utt_texts = model.utt_texts
    self.device = model.device

    tokenizer = processor.tokenizer
    cls_token_segment_id = 0
    sequence_a_segment_id = 0 if args.model_type in ['roberta', 'large'] else 1
    processor.special = { 'tokens': [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token],
      'ids': [cls_token_segment_id, sequence_a_segment_id, 0], 'maximum': [97, 100] }

    self.processor = processor
    self.tokenizer = tokenizer

    self.intent_list = processor.intent_labels
    self.action_list = processor.action_labels
    self.value_list = processor.value_labels
    self.enumerable_size = len(self.value_list)

    self.scenario_df = pd.read_csv(f'data/scenarios_0525.csv')
    ontology = json.load( open("data/ontology.json", "r") )
    self.non_enumerable = ontology["values"]["non_enumerable"]
    self.so_far = []  # hold a list of context utterances
    self.action_taken = False

    kb = json.load( open("data/kb.json", "r"))
    action_mask_map, intent_mask_map = Application.prepare_masks(kb, ontology)
    self.action_mask_map = action_mask_map
    self.intent_mask_map = intent_mask_map

  @staticmethod
  def prepare_masks(kb, ont):
    # record the range that needs to be masked out
    val_group_to_range = {}
    current_idx = 0
    before_cc = True
    num_enumerable_vals = 0
    # print(val_group_to_range)
    for val_group, values in ont["values"]["enumerable"].items():
        start = current_idx

        size = len(values)
        if 'credit card' in values:
            if before_cc:
                before_cc = False
            else:
                size -= 1

        num_enumerable_vals += size
        stop = start + size
        val_group_to_range[val_group] = (start, stop)
        current_idx = stop

    # build out the action to values mapping
    action_mask_map = {}
    for category, acts in ont['actions'].items():
        for action, values in acts.items():
            mask = np.zeros(100 + num_enumerable_vals)
            mask[num_enumerable_vals:] = 1.0

            for val_group in values:
                if val_group in val_group_to_range:
                    start, stop = val_group_to_range[val_group]
                    mask[start:stop] = 1.0

            action_mask_map[action] = mask

    # recreate the exact breakdown order from the loader
    options = []
    for section, buttons in ont["actions"].items():
      actions = buttons.keys()
      options.extend(actions)
    # double check that all actions in the kb are valid
    match, error = 0, 0
    for intent, actions in kb.items():
        for action in actions:
            if action in options:
                pass
            else:
                print(action)
                pdb.set_trace()
            # assert(action in options)
    # make the reverse lookup for the id that needs to be masked out
    action_to_idx = {action: index for index, action in enumerate(options)}
    # create the actual intent to action mapping
    intent_mask_map = {}
    for flow, subflows in ont['intents']['subflows'].items():
        for intent in subflows:
            mask = np.zeros(30)

            valid_options = kb[intent]
            for action in valid_options:
                mask[action_to_idx[action]] = 1.0

            intent_mask_map[intent] = mask

    return action_mask_map, intent_mask_map

  def delexicalize_text(self, scene, conversation):
    """ Given all the utterances within a converation and the scenario, delexicalize the 
    non enumerable entities. Inputs:
      - scene: a dict with detail, personal_info and order info
      - conversation: a list of utterances tuples where each tuple is (speaker, text, action, pred)
    Returns:
      - delex: a list of utterances where the text has been delexicalized
    """
    non_enumerable = []

    for slot in self.non_enumerable['personal']:
      if slot in scene:
        non_enumerable.append((slot, scene[slot]))

    for slot, value in scene.items():
      string_val = str(value)
      if string_val.startswith('$'):
        non_enumerable.append(('amount', string_val[1:]))
      if slot == 'order_id':
        non_enumerable.append((slot, scene[slot]))

    address = scene['address']
    address_tokens = address.split()
    address_halves = address.split(',')
    non_enumerable.append(('street_address', address_halves[0]))
    non_enumerable.append(('full_address', address[0]))
    non_enumerable.append(('zip_code', address_tokens[0]))

    delexed = []
    for utt in conversation:
      text = utt.replace('|', 'and').replace('_', ' ').lower()
      # must be in this order to prevent clash
      for slot, value in non_enumerable:
        if str(value) in text:
          text = text.replace(str(value), f'<{slot}>')

      delexed.append(text)
    return delexed

  def sample_scenario(self):
    scenario = self.scenario_df.sample()
    flow_detail = json.loads(scenario['Detail'].item())
    scene = json.loads(scenario['Personal'].item())  # default scene to the personal info

    order = json.loads(scenario['Order'].item())
    street_address = order['address']
    scene['address'] = f"{street_address} {order['city']}, {order['state']} {order['zip_code']}"

    for key, value in order.items():
      if key == 'products':
        for product in order['products']:
          product_name = product['brand'] + ' ' + product['product_type']
          scene[product_name] = '$' + str(product['amount'])
      if key not in ['address', 'city', 'status', 'zip_code', 'products']:
        scene[key] = value
    self.scene = scene

    issue = flow_detail['issue']
    reason = flow_detail['reason']
    solution = flow_detail['solution']
    prefix = flow_detail.get('prefix', 'Y')
    suffix = flow_detail.get('suffix', '')
    prompt = f"{prefix}ou {issue} because {reason}. Explain your problem to the agent, provide any information that is requested and attempt to {solution}. {suffix}"""

    return scene, prompt

  def take_action(self, intent_pred, action_pred, value_pred, context_tokens):
    top_intent = np.argmax(intent_pred)
    intent_name = self.intent_list[top_intent]

    # each intent mask should be size of 30 long
    intent_mask = self.intent_mask_map[intent_name]
    # now, all non valid actions should go to zero
    action_pred *= np.array(intent_mask)
    top_action = np.argmax(action_pred)
    action_name = self.action_list[top_action]
    
    # each action mask should be size of 223 long
    action_mask = self.action_mask_map[action_name]
    # now, all non valid values should go to zero
    value_pred *= np.array(action_mask)
    top_value = np.argmax(value_pred)
    if top_value < self.enumerable_size:  # part of enumerable
      value_name = self.value_list[top_value]
    else:                                 # copy from context
      top_value -= self.enumerable_size
      while top_value > len(context_tokens):
          top_value -= len(context_tokens)
      value_name = context_tokens[top_value]

    return {'Intent': intent_name, 'Action': action_name, 'Value': value_name}
