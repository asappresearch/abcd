import os, sys, pdb
import random
import math
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
    n_gpu = 0  # set the default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu > 0:   # this is not an 'else' statement and cannot be combined
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args

def check_cache(args, cache_dir):
  cache_filename = f"{args.model_type}_{args.task}"
  if args.cascade:
    cache_filename += "_cascade"
  cache_path = os.path.join(cache_dir, cache_filename)

  if os.path.exists(cache_path):
    print(f"Loading features from cached file {cache_path}")
    return cache_path, True
  else:
    print(f"Loading raw data and preparing new features")
    return cache_path, False

def check_directories(args):
    cache_dir = os.path.join(args.input_dir, 'cache')
    checkpoint_folder = f'{args.prefix}_{args.filename}_{args.model_type}_{args.suffix}'
    ckpt_dir = os.path.join(args.output_dir, args.task, checkpoint_folder)
    directories = [args.input_dir, cache_dir, args.output_dir, ckpt_dir]

    for directory in directories:
        if os.path.exists(directory):
            if directory == ckpt_dir:
                print(f"Warning: {directory} exists and files may be overwritten")
        else: 
            print(f"Creating {directory} directory ...")
            os.mkdir(directory)

    cache_results = check_cache(args, cache_dir)
    return ckpt_dir, cache_results

def prepare_inputs(batch, task, cascade, speaker_turn=False):

    if task == 'ast':
        full_history = {'input_ids': batch[0], 'token_type_ids': batch[1], 'attention_mask': batch[2]}
        targets = [batch[6], batch[7]] # actions and values
        context_tokens = {'input_ids': batch[3], 'token_type_ids': batch[4], 'attention_mask': batch[5]}
        tools, masks = None, None
        return full_history, targets, context_tokens, tools, masks
    else:
        full_history = {'input_ids': batch[0], 'token_type_ids': batch[1], 'attention_mask': batch[2]}
        candidates = batch[3]
        context_tokens = {'input_ids': batch[4], 'token_type_ids': batch[5], 'attention_mask': batch[6]}
        #           intent   nextstep   action    value     utterance
        targets = [batch[7], batch[8], batch[9], batch[10], batch[11]]
        masks = {'action': batch[12], 'value': batch[13], 'utterance': batch[14]}
        if cascade:
            targets.append(batch[15])
            targets.append(batch[16])
        if task == 'tcwi':
          tools = candidates, targets[0]
        else:
          tools = candidates
        return full_history, targets, context_tokens, tools, masks