from simple_parsing import ArgumentParser
from simple_parsing.helpers import choice, field
from dataclasses import dataclass


@dataclass
class BaseConfig:
    # Random seed
    seed: int = 14
    # Which type of encoder and tokenizer to use
    model_type: str = choice("roberta", "bert", "dialogpt", "albert", default="bert")
    # Choose which of the two major tasks to train the model
    task: str = choice("ast", "cds", default="ast")
    # Whether or not to go into debug mode, which is faster
    debug: bool = False
    # whether or not to have verbose prints
    verbose: bool = False


@dataclass
class DirectoryAndSavingConfig:
    """ ------ DIRECTORY AND SAVING -------- """
    output_dir: str = "outputs/"
    input_dir: str = "data"
    # distinguish the trial run, often a MM/DD date
    prefix: str = "0524"
    # name of the model if saving, or loading from saved
    filename: str = ""
    # distinguish the saved data, often a version number
    suffix: str = "v1"
    # Filter for just errors during evaluation
    filter: bool = False


@dataclass
class TrainingAndEvaluationConfig:
    """ ------ TRAINING AND EVALUATION -------- """
    # load the best saved model and run evaluation, qualify or quantify flags must be on
    do_eval: bool = False
    log_interval: int = 100
    # examine the qualitative outputs of the model in natural language
    qualify: bool = False
    # examine the quantitative outputs of the model in reports
    quantify: bool = False


@dataclass
class MajorModelOptions:
    """ ------- MAJOR MODEL OPTIONS -------- """
    # use cascading evaluation rather than turn level
    cascade: bool = False
    # use an oracle intent classification module
    use_intent: bool = False
    # take advantage of KB guidelines to limit action and value options
    use_kb: bool = False


@dataclass
class DatasetCreation:
    """ ------ DATASET CREATION -------- """
    # which version of the dataset is being used
    # v1.0 was used initially, but v1.1 is released as a significantly cleaner benchmark
    version: float = 1.1
    # whether to build new vocabulary of Glove vectors
    build_vocab: bool = False
    # Maximum number of tokens to truncate each utterance
    max_seq_len: int = 512


@dataclass
class ParameterOptimizationConfig:
    """ hyperparameters """
    # use RAdam optimizer rather than default AdamW
    radam: bool = False
    # Learning rate alpha for weight updates
    learning_rate: float = field(default=3e-5, alias="-lr")
    # Number of hidden units, size of hidden dimension
    hidden_dim: int = 768
    # probability of dropping a node, opposite of keep prob
    drop_prob: float = 0.2
    # Number of steps for gradient accumulation
    grad_accum_steps: int = 1
    # weight_decay to regularize the weights
    weight_decay: float = field(default=0.003, alias="-reg")
    # batch size for training and evaluation
    batch_size: int = 50
    # Number of epochs or episodes to train
    epochs: int = field(default=14, alias="-e")


@dataclass
class Config(
    BaseConfig,
    DirectoryAndSavingConfig,
    TrainingAndEvaluationConfig,
    MajorModelOptions,
    DatasetCreation,
    ParameterOptimizationConfig,    
):
    """ Parameters for the main.py script. """
    pass


def solicit_params() -> Config:
    parser = ArgumentParser(add_option_string_dash_variants=True)
    parser.add_arguments(Config, "config")
    args = parser.parse_args()
    config: Config = args.config
    return args.config
    
    
    # parser.add_argument("--seed", help="Random seed", type=int, default=14)
    # parser.add_argument(
    #     "--model-type",
    #     choices=["roberta", "bert", "dialogpt", "albert"],
    #     help="Which type of encoder and tokenizer to use",
    #     default="bert",
    # )
    # parser.add_argument(
    #     "--task",
    #     default="ast",
    #     type=str,
    #     choices=["ast", "cds"],
    #     help="choose which of the two major tasks to train the model",
    # )
    # parser.add_argument(
    #     "--debug",
    #     default=False,
    #     action="store_true",
    #     help="whether or not to go into debug mode, which is faster",
    # )
    # parser.add_argument(
    #     "-v",
    #     "--verbose",
    #     default=False,
    #     action="store_true",
    #     help="whether or not to have verbose prints",
    # )

    # # ------ DIRECTORY AND SAVING --------
    # parser.add_argument("--output-dir", default="outputs/", type=str)
    # parser.add_argument("--input-dir", default="data/", type=str)
    # parser.add_argument(
    #     "--prefix",
    #     type=str,
    #     default="0524",
    #     help="distinguish the trial run, often a MM/DD date",
    # )
    # parser.add_argument(
    #     "--filename",
    #     type=str,
    #     help="name of the model if saving, or loading from saved",
    # )
    # parser.add_argument(
    #     "--suffix",
    #     type=str,
    #     default="v1",
    #     help="distinguish the saved data, often a version number",
    # )
    # parser.add_argument(
    #     "--filter",
    #     default=False,
    #     action="store_true",
    #     help="Filter for just errors during evaluation",
    # )

    # # ------ TRAINING AND EVALUATION --------
    # parser.add_argument(
    #     "--do-eval",
    #     default=False,
    #     action="store_true",
    #     help="load the best saved model and run evaluation, qualify or quantify flags must be on",
    # )
    # parser.add_argument("--log-interval", default=100, type=int)
    # parser.add_argument(
    #     "--qualify",
    #     default=False,
    #     action="store_true",
    #     help="examine the qualitative outputs of the model in natural language",
    # )
    # parser.add_argument(
    #     "--quantify",
    #     default=False,
    #     action="store_true",
    #     help="examine the quantitative outputs of the model in reports",
    # )

    # # ------- MAJOR MODEL OPTIONS --------
    # parser.add_argument(
    #     "--cascade",
    #     default=False,
    #     action="store_true",
    #     help="use cascading evaluation rather than turn level",
    # )
    # parser.add_argument(
    #     "--use-intent",
    #     default=False,
    #     action="store_true",
    #     help="use an oracle intent classification module",
    # )
    # parser.add_argument(
    #     "--use-kb",
    #     default=False,
    #     action="store_true",
    #     help="take advantage of KB guidelines to limit action and value options",
    # )

    # # ------ DATASET CREATION --------
    # parser.add_argument(
    #     "--version",
    #     type=float,
    #     default=1.1,
    #     help="which version of the dataset is being used",
    # )
    # # v1.0 was used initially, but v1.1 is released as a significantly cleaner benchmark
    # parser.add_argument(
    #     "--build-vocab",
    #     default=False,
    #     action="store_true",
    #     help="whether to build new vocabulary of Glove vectors",
    # )
    # parser.add_argument(
    #     "--max-seq-len",
    #     default=512,
    #     type=int,
    #     help="Maximum number of tokens to truncate each utterance",
    # )

    # # ------ PARAMETER OPTIMIZATION --------
    # param_group = parser.add_argument_group(title="hyperparameters")
    # parser.add_argument(
    #     "--radam",
    #     default=False,
    #     action="store_true",
    #     help="use RAdam optimizer rather than default AdamW",
    # )
    # param_group.add_argument(
    #     "-lr",
    #     "--learning-rate",
    #     default=3e-5,
    #     type=float,
    #     help="Learning rate alpha for weight updates",
    # )
    # param_group.add_argument(
    #     "--hidden-dim",
    #     default=768,
    #     type=int,
    #     help="Number of hidden units, size of hidden dimension",
    # )
    # param_group.add_argument(
    #     "--drop-prob",
    #     default=0.2,
    #     type=float,
    #     help="probability of dropping a node, opposite of keep prob",
    # )
    # param_group.add_argument(
    #     "--grad-accum-steps",
    #     default=1,
    #     type=int,
    #     help="Number of steps for gradient accumulation",
    # )
    # param_group.add_argument(
    #     "-reg",
    #     "--weight-decay",
    #     default=0.003,
    #     type=float,
    #     help="weight_decay to regularize the weights",
    # )
    # param_group.add_argument(
    #     "--batch-size",
    #     default=50,
    #     type=int,
    #     help="batch size for training and evaluation",
    # )
    # param_group.add_argument(
    #     "-e",
    #     "--epochs",
    #     default=14,
    #     type=int,
    #     help="Number of epochs or episodes to train",
    # )

    args = parser.parse_args()
    return args
