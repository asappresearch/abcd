""" TypedDict for the Ontology dict and its items. """
from typing import List, Any, Union, Dict

try:
    from typing import TypedDict, Literal
except ImportError:
    from typing_extensions import TypedDict, Literal  # type: ignore


class Intents(TypedDict):
    flows: List[str]
    subflows: Dict[str, List[str]]


class Vocabulary(TypedDict):
    tokens: List
    # Special tokens: ["[CLS]", "[SEP]", "[UNK]", "[AGENT]", "[CUSTOMER]", "[ACTION]"]
    special: List[str]


# Action maps the action name to the list of values that need to be entered, e.g:
# "validate-purchase": ["username", "email", "order_id"],
Action = Dict[str, List[str]]

# "Actions" dicts apparently map the 'section' to the 'buttons' in that section, e.g.:
# "kb_query": {
#         "verify-identity": ["customer_name", "account_id", "order_id", "zip_code"],
#         ...
# }
Actions = Dict[str, Action]


class Values(TypedDict):
    enumerable: Dict[str, List[str]]
    non_enumerable: Dict[str, List[str]]


class Ontology(TypedDict):
    """ TypedDict for the Ontology, which is in `data/ontology.json`
    """

    intents: Intents
    vocabulary: Vocabulary
    actions: Actions
    values: Values
    next_steps: List[str]  # ["retrieve_utterance", "take_action", "end_conversation"]
