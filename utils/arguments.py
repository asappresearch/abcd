from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def solicit_params():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('--seed', help='Random seed', type=int, default=14)
  parser.add_argument('--model-type', choices=['roberta', 'bert', 'dialogpt', 'albert'],
            help='Which type of encoder and tokenizer to use', default='bert')
  parser.add_argument('--task', default='ast', type=str, choices=['ast', 'cds'],
            help='choose which of the two major tasks to train the model', )
  parser.add_argument('--debug', default=False, action='store_true',
            help='whether or not to go into debug mode, which is faster')
  parser.add_argument('-v', '--verbose', default=False, action='store_true',
            help='whether or not to have verbose prints')

  # ------ DIRECTORY AND SAVING --------
  parser.add_argument('--output-dir', default='outputs/', type=str)
  parser.add_argument('--input-dir', default='data/', type=str)
  parser.add_argument('--prefix', type=str, default='0524',
            help='distinguish the trial run, often a MM/DD date')
  parser.add_argument('--filename', type=str, 
            help='name of the model if saving, or loading from saved')
  parser.add_argument('--suffix', type=str, default='v1',
            help='distinguish the saved data, often a version number')
  parser.add_argument('--filter', default=False, action='store_true',
            help='Filter for just errors during evaluation')

  # ------ TRAINING AND EVALUATION --------
  parser.add_argument('--do-eval', default=False, action='store_true',
            help='load the best saved model and run evaluation, qualify or quantify flags must be on')
  parser.add_argument('--log-interval', default=100, type=int)
  parser.add_argument('--qualify', default=False, action='store_true',
            help='examine the qualitative outputs of the model in natural language')
  parser.add_argument('--quantify', default=False, action='store_true',
            help='examine the quantitative outputs of the model in reports')

  # ------- MAJOR MODEL OPTIONS --------
  parser.add_argument('--cascade', default=False, action='store_true',
            help='use cascading evaluation rather than turn level')
  parser.add_argument('--use-intent', default=False, action='store_true',
            help='use an oracle intent classification module')
  parser.add_argument('--use-kb', default=False, action='store_true',
            help='take advantage of KB guidelines to limit action and value options')

  # ------ DATASET CREATION --------
  parser.add_argument('--version', type=float, default=1.1,
            help="which version of the dataset is being used")
            # v1.0 was used initially, but v1.1 is released as a significantly cleaner benchmark
  parser.add_argument('--build-vocab', default=False, action='store_true',
            help='whether to build new vocabulary of Glove vectors')
  parser.add_argument('--max-seq-len', default=512, type=int,
            help='Maximum number of tokens to truncate each utterance')

  # ------ PARAMETER OPTIMIZATION --------
  param_group = parser.add_argument_group(title='hyperparameters')
  parser.add_argument('--radam', default=False, action='store_true',
            help='use RAdam optimizer rather than default AdamW')
  param_group.add_argument('-lr', '--learning-rate', default=3e-5, type=float,
            help='Learning rate alpha for weight updates')
  param_group.add_argument('--hidden-dim', default=768, type=int,
            help='Number of hidden units, size of hidden dimension')
  param_group.add_argument('--drop-prob', default=0.2, type=float,
            help='probability of dropping a node, opposite of keep prob')
  param_group.add_argument('--grad-accum-steps', default=1, type=int,
            help='Number of steps for gradient accumulation')
  param_group.add_argument('-reg', '--weight-decay', default=0.003, type=float,
            help='weight_decay to regularize the weights')
  param_group.add_argument('--batch-size', default=50, type=int,
            help='batch size for training and evaluation')
  param_group.add_argument('-e', '--epochs', default=14, type=int,
            help='Number of epochs or episodes to train')

  args = parser.parse_args()
  return args

