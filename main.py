import os, sys, pdb
import random
import numpy as np
import torch
from tqdm import tqdm as progress_bar
from typing import Dict, Union

from abcd.utils.arguments import solicit_params, Config
from abcd.utils.help import set_seed, setup_gpus, check_directories, prepare_inputs, device
from abcd.utils.load import (
    load_data,
    load_tokenizer,
    load_candidates,
    get_optimizer,
    get_scheduler,
)
from abcd.utils.process import process_data, setup_dataloader
from abcd.utils.evaluate import quantify, qualify

from abcd.components.datasets import ActionDataset, CascadeDataset
from abcd.components.tools import ExperienceLogger
from abcd.components.models import ActionStateTracking, CascadeDialogSuccess


def run_main(
    args: Config,
    datasets: Dict[str, Union[ActionDataset, CascadeDataset]],
    model: Union[ActionStateTracking, CascadeDialogSuccess],
    exp_logger: ExperienceLogger,
):
    if args.task == "cds":
        utt_data = load_candidates(args)
        model.add_candidate_data(*utt_data)
    kb_labels = {}
    if args.use_kb:
        kb_labels["intent"] = list(model.mappings["intent"].keys())
        kb_labels["action"] = list(model.mappings["action"].keys())

    exp_logger.init_tb_writers()
    run_train(args, datasets, model, exp_logger, kb_labels)

    if args.do_eval:
        result = run_eval(args, datasets, model, exp_logger, kb_labels, split="test")
        results = dict((k + f"_{args.filename}", v) for k, v in result.items())
        print("Test Results -", results)


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

    intent_loss = loss_func(intent_scores, intent_target)
    nextstep_loss = loss_func(nextstep_scores, nextstep_target)
    action_loss = loss_func(action_scores, action_target)
    value_loss = loss_func(value_scores, value_target)

    utt_target_ids = utterance_target.unsqueeze(1)  # batch_size, 1
    chosen = torch.gather(utt_scores, dim=1, index=utt_target_ids)
    correct = chosen.sum()  # scalar

    shift = torch.max(utt_scores)  # perform log sum exp of the incorrect scores
    res = torch.exp(utt_scores - shift)  # batch_size, num_candidates
    res = torch.log(torch.sum(res, dim=1))  # batch_size
    incorrect = torch.sum(
        shift + res
    )  # add the shift back in to complete the log-sum-exp overflow trick
    utt_loss = incorrect - correct

    total_loss = intent_loss + nextstep_loss + action_loss + value_loss + utt_loss
    return total_loss


def run_train(args, datasets, model, exp_logger, kb_labels):
    dataloader, num_examples = setup_dataloader(
        datasets, args.batch_size, split="train"
    )
    t_total = len(dataloader) // args.grad_accum_steps * args.epochs
    exp_logger.start_train(num_examples, total_step=t_total)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer, t_total)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    model.zero_grad()

    for epoch in range(args.epochs):
        model.train()

        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)

            if args.task == "ast":
                full_history, targets, context_tokens, _ = prepare_inputs(args, batch)
                scores = model(full_history, context_tokens)
                loss = ast_loss(scores, targets, loss_func)
            elif args.task == "cds":
                full_history, targets, context_tokens, tools = prepare_inputs(
                    args, batch
                )
                scores = model(full_history, context_tokens, tools)
                loss = cds_loss(scores, targets, loss_func)

            if args.grad_accum_steps > 1:
                loss = loss / args.grad_accum_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                result, metric = quantify(args, scores, targets, "train")
                exp_logger.log_train(step, loss.item(), result, metric)

            if args.debug and step > 3 * args.log_interval:
                break

        result, res_name = run_eval(
            args, datasets, model, exp_logger, kb_labels, split="dev"
        )
        dev_score = result[res_name]
        if dev_score > exp_logger.best_score:
            model.save_pretrained(exp_logger.filepath)
            exp_logger.best_score = dev_score
        exp_logger.log_dev(step + 1, res_name, dev_score)


def run_eval(args, datasets, model, exp_logger, kb_labels, split="dev"):
    dataloader, num_examples = setup_dataloader(datasets, args.batch_size, split)
    exp_logger.start_eval(num_examples, kind=args.filename)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    num_outputs = len(model.outputs)
    model.eval()

    preds, labels, convo_ids, turn_counts = [], [], [], []
    for batch in progress_bar(
        dataloader, total=len(dataloader), desc=f"Epoch {exp_logger.epoch}"
    ):
        batch = tuple(t.to(device) for t in batch)
        full_history, batch_targets, context_tokens, tools = prepare_inputs(args, batch)

        with torch.no_grad():
            if args.task == "ast":
                batch_scores = model(full_history, context_tokens)
                batch_loss = ast_loss(batch_scores, batch_targets, loss_func)
            elif args.task == "cds":
                batch_scores = model(full_history, context_tokens, tools)
                batch_loss = cds_loss(batch_scores, batch_targets, loss_func)

        if args.cascade:
            batch_turn_count = batch_targets.pop()
            batch_convo_id = batch_targets.pop()

        if args.quantify or split == "dev":
            exp_logger.eval_loss += batch_loss.mean().item()
            exp_logger.batch_steps += 1

        preds.append(batch_scores)
        labels.append(batch_targets)
        convo_ids.append(batch_convo_id if args.cascade else 0)
        turn_counts.append(batch_turn_count if args.cascade else 0)

        if args.debug:
            if len(turn_counts) > 10:
                break

    grouped_preds = [
        torch.cat([pred[i] for pred in preds], dim=0) for i in range(num_outputs)
    ]
    grouped_labels = [
        torch.cat([label[i] for label in labels], dim=0) for i in range(num_outputs)
    ]
    ci_and_tc = (
        (torch.cat(convo_ids, dim=0), torch.cat(turn_counts, dim=0))
        if args.cascade
        else (0, 0)
    )

    utils = {"kb_labels": kb_labels, "ci_and_tc": ci_and_tc}
    metrics, res_name = quantify(args, grouped_preds, grouped_labels, utils)
    exp_logger.end_eval(metrics, kind=args.filename)
    return (metrics, res_name) if split == "dev" else metrics


if __name__ == "__main__":
    args = solicit_params()
    args = setup_gpus(args)
    set_seed(args)

    ckpt_dir, cache_results = check_directories(args)
    raw_data = load_data(args, cache_results[1])
    tokenizer, ontology = load_tokenizer(args)
    features, mappings = process_data(
        args, tokenizer, ontology, raw_data, *cache_results
    )
    exp_logger = ExperienceLogger(args, ckpt_dir)

    datasets: Dict[str, Union[ActionDataset, CascadeDataset]]
    model: Union[ActionStateTracking, CascadeDialogSuccess]

    if args.task == "ast":
        datasets = {
            split: ActionDataset(args, feats) for split, feats in features.items()
        }
        model = ActionStateTracking(args, mappings, ckpt_dir)
    elif args.task == "cds":
        datasets = {
            split: CascadeDataset(args, feats) for split, feats in features.items()
        }
        model = CascadeDialogSuccess(args, mappings, ckpt_dir)

    model = model.to(device)
    model.encoder.resize_token_embeddings(len(tokenizer))
    run_main(args, datasets, model, exp_logger)
