import os, sys, pdb
import random
import numpy as np
import torch
from tqdm import tqdm as progress_bar

from utils.arguments import solicit_params
from utils.help import set_seed, setup_gpus, check_directories, prepare_inputs, device
from utils.load import load_data, load_tokenizer, load_candidates, get_optimizer, get_scheduler
from utils.process import process_data, setup_dataloader
from utils.evaluate import quantify, qualify

from components.datasets import ActionDataset, CascadeDataset
from components.tools import ExperienceLogger
from components.models import ActionStateTracking, CascadeDialogSuccess

def run_main(args, datasets, model, exp_logger):
  if args.task == 'cds':
    utt_data = load_candidates(args)
    model.add_candidate_data(*utt_data)
  kb_labels = {'intent': dataset.intent_labels, 'action': dataset.action_labels} if args.use_kb else {}
  exp_logger.init_tb_writers()
  run_train(args, datasets, model, exp_logger, kb_labels)

  if args.do_eval:
    result = run_eval(args, datasets['test'], model, exp_logger, kb_labels, split='test')
    results = dict((k + f'_{args.filename}', v) for k, v in result.items())
    print(results)

def ast_loss(scores, targets, loss_func):
  action_score, value_score = scores
  action_target, value_target = targets

  action_loss = loss_func(action_score, action_target)
  value_loss = loss_func(value_score, value_target)

  total_loss = action_loss + value_loss
  return total_loss

def cds_loss(scores, targets, loss_func):
  intent_scores, nextstep_scores, action_scores, value_scores, utt_scores = scores
  intent_target, nextstep_target, action_target, value_target, utt_target = targets
  
  utterance_mask = nextstep_target == 0  # 0 is the index of 'retrieve_utterance'
  batch_size, num_candidates = utt_scores.shape
  utt_scores = utt_scores * utterance_mask.unsqueeze(1).repeat(1, num_candidates)
  utterance_target = utt_target * utterance_mask

  intent_loss   = loss_func(intent_scores, intent_target)
  nextstep_loss = loss_func(nextstep_scores, nextstep_target)
  action_loss   = loss_func(action_scores, action_target)
  value_loss    = loss_func(value_scores, value_target)
  
  utt_target_ids = utterance_target.unsqueeze(1)   # batch_size, 1
  chosen = torch.gather(utt_scores, dim=1, index=utt_target_ids)
  correct = chosen.sum()                   # scalar

  shift = torch.max(utt_scores)             # perform log sum exp of the incorrect scores
  res = torch.exp(utt_scores - shift)       # batch_size, num_candidates
  res = torch.log(torch.sum(res, dim=1))   # batch_size
  incorrect = torch.sum(shift + res)       # add the shift back in to complete the log-sum-exp overflow trick
  utt_loss = incorrect - correct

  total_loss = intent_loss + nextstep_loss + action_loss + value_loss + utt_loss
  return total_loss

def run_train(args, datasets, model, exp_logger, kb_labels):
  dataloader, num_examples = setup_dataloader(datasets, args.batch_size, split='train')
  t_total = len(dataloader) // args.grad_accum_steps * args.epochs    
  exp_logger.start_train(num_examples, total_step=t_total)
  optimizer = get_optimizer(args, model)
  scheduler = get_scheduler(args, optimizer, t_total)
  model.zero_grad()

  for epoch in range(args.epochs):
    model.train()

    for step, batch in enumerate(dataloader):
      batch = tuple(t.to(device) for t in batch)
      loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)

      if args.task == 'ast':
        full_history, targets, context_tokens, _ = prepare_inputs(args, batch)
        scores = model(full_history, context_tokens)
        loss = ast_loss(scores, targets, loss_func)
      elif args.task == 'cds':
        full_history, targets, context_tokens, tools = prepare_inputs(args, batch)
        scores = model(full_history, context_tokens, tools)
        loss = cds_loss(scores, targets, loss_func)

      if args.grad_accum_steps > 1:
        loss = loss / args.grad_accum_steps
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
          result, res_name, eval_loss = run_eval(args, datasets, model, exp_logger, kb_labels, split='dev')
          dev_score = result[res_name]
          exp_logger.log_dev(model, step+1, eval_loss, res_name, dev_score)

      if args.debug:
        print("step", step)
        if step == 3: sys.exit()

def run_eval(args, datasets, model, exp_logger, kb_labels, split='dev'):
  dataloader, num_examples = setup_dataloader(datasets, args.batch_size, split)  
  exp_logger.start_eval(num_examples, kind=args.filename)
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
    full_history, batch_targets, context_tokens, tools = prepare_inputs(args, batch)

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
    datasets = {split: ActionDataset(args, feats) for split, feats in features.items()}
    model = ActionStateTracking(args, mappings, ckpt_dir)
  elif args.task == 'cds':
    datasets = {split: CascadeDataset(args, feats) for split, feats in features.items()}
    model = CascadeDialogSuccess(args, mappings, ckpt_dir)

  model = model.to(device)
  model.encoder.resize_token_embeddings(len(tokenizer))
  run_main(args, datasets, model, exp_logger)