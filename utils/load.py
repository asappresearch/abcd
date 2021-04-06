import os, sys, pdb
import csv
import json
import random
import numpy as np

from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer
    
def load_data(args, already_cached):
  if already_cached:
    return []  # no need to load raw_data since we already have a feature cache
  else:
    data_path = os.path.join(args.input_dir, f"abcd_v{args.version}.json")
    raw_data = json.load(open(data_path, 'r'))
    return raw_data

def load_tokenizer(args):
  ontology = json.load(open(f'{args.input_dir}/ontology.json', 'r'))
  non_enumerable = ontology['values']['non_enumerable']
  special = [f'<{slot}>' for category, slots in non_enumerable.items() for slot in slots]

  if args.model_type == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  elif args.model_type == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  elif args.model_type == 'albert':
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

  tokenizer.add_tokens(special)
  return tokenizer, ontology

def get_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    print("Using special optimizer")
    optimizer = RAdam(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def get_scheduler(args, optimizer, training_steps):
    if args.warmup_steps == 0:  # use the warmup ratio instead
        warmup_steps = math.ceil(training_steps * args.warmup_ratio)
    else:
        warmup_steps = args.warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)
    return scheduler