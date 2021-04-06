import os, sys, pdb
import random
import numpy as np
import torch
from tqdm import tqdm as progress_bar
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader as DataSampler

from utils.arguments import solicit_params
from utils.help import set_seed, setup_gpus, check_directories, prepare_inputs, device
from utils.load import load_data, load_tokenizer
from utils.process import process_data
from utils.evaluate import quantify, qualify

from components.datasets import ActionDataset, CascadeDataset
from components.tools import ExperienceLogger, RAdamOptimizer
from components.models import ActionStateTracking, CascadeDialogSuccess

def run_main(args, dataset, model, exp_logger):
  """ if dataset.chunk_count['train'] > 1: exp_logger.init_tb_writers()
    run_secondary(args, dataset, model, exp_logger)"""
  if args.task == 'cds':
    model.add_candidate_tooling(dataset)
  kb_labels = {'intent': dataset.intent_labels, 'action': dataset.action_labels} if args.use_kb else {}

  if args.do_train:
    exp_logger.init_tb_writers()
    run_train(args, dataset, model, exp_logger, kb_labels)

  if args.do_eval:
    result = run_eval(args, dataset, model, exp_logger, kb_labels, split='test')
    results = dict((k + f'_{args.filename}', v) for k, v in result.items())
    print(results)

def run_train(args, dataset, model, exp_logger, kb_labels):
  train_dataset = dataset.train
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataSampler(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

  t_total = len(train_dataloader) // args.grad_accum_steps * args.epochs    
  exp_logger.start_train(num_examples=len(train_dataset), total_step=t_total)
  optimizer = get_optimizer(args, model)
  scheduler = get_scheduler(args, optimizer, t_total)
  model.zero_grad()

  for epoch in range(args.epochs):
    model.train()

    for step, batch in enumerate(train_dataloader):
      batch = tuple(t.to(device) for t in batch)

      if args.task == 'ast':
        full_history, targets, context_tokens = prepare_inputs(batch, args.task, args.cascade)
        scores = model(full_history, context_tokens)
        loss_func = get_loss(args.task)
        loss = loss_func(scores, targets)
      elif args.task == 'cds':
        full_history, targets, context_tokens, candidates, masks = prepare_inputs(batch, args.task, args.cascade)
        scores = model(full_history, context_tokens, candidates)
        loss_func = get_loss(args.task)
        loss = loss_func(scores, targets, masks)
      # elif args.task == 'tcwi':
      # full_history, targets, context_tokens, candidates, masks = prepare_inputs(batch, args.task, args.cascade)
      # scores = model(full_history, context_tokens, candidates, targets[0])
      # loss_func = get_loss(args.task)
      # loss = loss_func(scores, targets, masks)

      if args.grad_accum_steps > 1:
        loss = loss / args.grad_accum_steps

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

      if (step+1) % args.grad_accum_steps == 0:
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        exp_logger.global_step += 1

        if args.log_interval > 0 and exp_logger.global_step % args.log_interval == 0:
          loss_result = loss.item()
          exp_logger.log_train(step+1, loss_result, scores, targets)

        if args.eval_interval > 0 and exp_logger.global_step % args.eval_interval == 0:
          # Log eval stats and if performs better than save checkpoint
          result, res_name, eval_loss = run_eval(args, dataset, model, exp_logger, kb_labels, split='dev')
          dev_score = result[res_name]
          exp_logger.log_dev(model, step+1, eval_loss, res_name, dev_score)

      if args.debug:
        print("step", step)
        if step == 2: sys.exit()

def run_eval(args, processor, model, exp_logger, kb_labels, split='dev'):
  dataset = processor.dev if split == 'dev' else processor.test
  eval_sampler = SequentialSampler(dataset)
  eval_dataloader = DataSampler(dataset, sampler=eval_sampler, batch_size=args.batch_size)
  
  exp_logger.start_eval(len(dataset), kind=args.filename)
  all_preds, all_labels = [], []
  convo_ids, turn_counts = [], []
  model.eval()

  if args.quantify or split=='dev':
    eval_loss, num_eval_steps = 0, 0
    num_outputs = len(model.outputs)
  if args.qualify or args.breakdown:
    tokenizer = processor.tokenizer
    target_maps = [processor.action_labels, processor.value_labels]
    if args.task == 'cds':
      target_maps.extend([processor.intent_labels, processor.nextstep_labels])
  
  for batch in progress_bar(eval_dataloader, desc="Evaluating"):
    batch = tuple(t.to(device) for t in batch)
    raw_inputs = batch, args.task, args.cascade
    full_history, batch_targets, context_tokens, tools, masks = prepare_inputs(*raw_inputs)

    with torch.no_grad():
      batch_scores = model(full_history, context_tokens, tools)
    loss_func = get_loss(args.task)
    if args.cascade and args.task in ['cds', 'tcwi']:
      batch_turn_count = batch_targets.pop()
      batch_convo_id = batch_targets.pop()

    if args.quantify or split=='dev':
      batch_eval_loss = loss_func(batch_scores, batch_targets, masks)
      eval_loss += batch_eval_loss.mean().item()
      num_eval_steps += 1
    elif args.qualify:
      history_ids = full_history['input_ids'].detach().cpu().numpy()
      context_ids = context_tokens['input_ids'].detach().cpu().numpy()
      ids = [history_ids, context_ids]
      qualify(args, ids, tokenizer, target_maps, batch_scores, batch_targets)
    
    all_preds.append(batch_scores)
    all_labels.append(batch_targets)
    convo_ids.append(batch_convo_id if args.cascade else 0)
    turn_counts.append(batch_turn_count if args.cascade else 0)
    
  grouped_preds = [torch.cat([pred[i] for pred in all_preds], dim=0) for i in range(num_outputs)]
  grouped_labels = [torch.cat([label[i] for label in all_labels], dim=0) for i in range(num_outputs)]
  preds_and_labels = [grouped_preds, grouped_labels]
  ci_and_tc = (torch.cat(convo_ids, dim=0), torch.cat(turn_counts, dim=0)) if args.cascade else (0, 0)

  eval_loss = eval_loss / num_eval_steps
  tools = {}
  if args.use_kb:   tools['kb_labels'] = kb_labels
  if args.cascade:  tools['ci_and_tc'] = ci_and_tc
  metrics, res_name = quantify(args, preds_and_labels, tools)
  exp_logger.end_eval(metrics, kind=args.filename)
  return (metrics, res_name, eval_loss) if split == 'dev' else metrics


if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  set_seed(args)

  ckpt_dir, cache_results = check_directories(args)
  raw_data = load_data(args, cache_results[1])
  tokenizer, ontology = load_tokenizer(args)
  features, mappings = process_data(args, tokenizer, ontology, raw_data, *cache_results)
  exp_logger = ExperienceLogger(args, ckpt_dir)

  if args.task == 'ast':
    datasets = [ActionDataset(args, feats, split) for split, feats in features.items()]
    model = ActionStateTracking(args, mappings, ckpt_dir)
  elif args.task == 'cds':
    datasets = [CascadeDataset(args, feats, split) for split, feats in features.items()]
    model = CascadeDialogSuccess(args, mappings, ckpt_dir)

  model = model.to(device)
  model.encoder.resize_token_embeddings(len(tokenizer))
  run_main(args, datasets, model, exp_logger)