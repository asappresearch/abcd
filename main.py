import os, sys, pdb
import random
import numpy as np
import torch
from tqdm import tqdm as progress_bar
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader as DataSampler

from loader import data_loaders
from utils.help import check_directories, get_optimizer, get_scheduler
from utils.evaluate import quantify, qualify, interact
from utils.preprocess import processors, prepare_inputs
from utils.arguments import solicit_params

from components.loggers import ExperienceLogger
from components.models import get_model, get_loss, get_tokenizer, device

def run_main(args, dataset, model, exp_logger):
  pass

def run_train(args, dataset, model, exp_logger, kb_labels):
  pass

def run_eval(args, processor, model, exp_logger, kb_labels, split='dev'):
  pass


if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  run_main(args, dataset, model, exp_logger)


