""" Simple set of typed dicts that match the data format of ABCD.

This doesn't have any runtime impact, it's just to make it easier to understand what 
the data contains in code.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union, NamedTuple, Dict, Mapping
from simple_parsing.helpers import JsonSerializable
try:
    from typing import TypedDict, Literal
except ImportError:
    from typing_extensions import TypedDict, Literal  # type: ignore


Speaker = Literal["customer", "agent", "action"]

# NOTE: This is still a bit unclear, but the values for this second field appear to be
# related to the next_steps in the ontology json.
NextStep = Literal["retrieve_utterance", "take_action", "end_conversation"]


class Targets(NamedTuple):
    """ Tuple of five items representing the subtask labels

    Examples:
    
    - When in an agent's turn:
    ```json
    "targets": [
        "return_size",
        "take_action",
        "validate-purchase",
        [
            "cminh730",
            "cminh730@email.com",
            "3348917502"
        ],
        -1,
    ]
    ```

    - When in a user's turn:
    ```json
    "targets": [
        "return_size",
        null,
        null,
        [],
        -1
    ]
    ```
    """
    #  Intent Classification (text) - 55 subflow options
    subflow_name: str
    # Nextstep Selection (text) - take_action, retrieve_utterance or end_conversation; 3 options
    next_step: Optional[NextStep]
    # Action Prediction (text) - the button clicked by the agent; 30 options
    action: Optional[str]
    # The values that are passed to the action.
    # Value Filling (list) - the slot value(s) associated with the action above; 125 options
    values: List[str]
    # Utterance Ranking (int) - target position within list of candidates; 100 options
    some_integer: int


class Turn(TypedDict):
    # Either "agent", "customer" or "action"
    speaker: Speaker
    # The utterance of the agent/customer or the system generated response of the action
    text: str
    # Integer representing the turn number, starting from 1
    turn_count: int
    #  Tuple of five items representing the subtask labels
    targets: Targets
    # List of utterance ids representing the pool of 100 candidates to choose from when ranking.
    # The surface form text can be found in `utterances.json` where the utt_id is the index.
    # Only applicable when the current turn is a "retrieve_utterance" step.
    candidates: List[int]


class Personal(TypedDict):
    customer_name: str
    email: str
    member_level: str
    phone: str
    username: str


class ProductItem(TypedDict):
    brand: str
    product_type: str
    amount: int
    image_url: str


class Order(TypedDict):
    street_address: str
    full_address: str
    city: str
    num_products: str
    order_id: str
    packaging: str
    payment_method: str
    products: List[ProductItem]
    purchase_date: str
    state: str
    zip_code: str


class ProductDict(TypedDict):
    names: List[str]
    amounts: List[int]


class Scenario(TypedDict):
    """ Typed Dict for the 'ccenario' field of a "conversation" in ABCD dataset. 
    
    Examples:
    ```json
    "scenario": {
        "personal": {
            "customer_name": "crystal minh",
            "email": "cminh730@email.com",
            "member_level": "bronze",
            "phone": "(977) 625-2661",
            "username": "cminh730"
        },
        "order": {
            "street_address": "6821 1st ave",
            "full_address": "6821 1st ave  san mateo, ny 75227",
            "city": "san mateo",
            "num_products": "1",
            "order_id": "3348917502",
            "packaging": "yes",
            "payment_method": "credit card",
            "products": "[{'brand': 'michael_kors', 'product_type': 'jeans', 'amount': 94, 'image_url': 'images/michael_kors-jeans.jpeg'}]",
            "purchase_date": "2019-11-06",
            "state": "ny",
            "zip_code": "75227"
        },
        "product": {
            "names": [
                "michael_kors jeans"
            ],
            "amounts": [
                94
            ]
        },
        "flow": "product_defect",
        "subflow": "return_size"
    }
    ```
    """
    personal: Personal
    order: Order
    product: ProductDict
    flow: str
    subflow: str


class DialogueItem(NamedTuple):
    speaker: Speaker
    text: str


class Conversation(TypedDict):
    convo_id: int
    scenario: Scenario
    original: List[DialogueItem]
    delexed: List[Turn]



Split = Literal["train", "dev", "test"]
# Could just use Dict[str, List[Conversation]] to be more general as well.
RawData = Dict[Split, List[Conversation]]

