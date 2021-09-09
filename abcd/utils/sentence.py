import gym
from textworld.gym.spaces import Word
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from english_words import english_words_set
from transformers import T5Tokenizer
from typing import Tuple, Any
from transformers import GPT2TokenizerFast
from transformers import BertPreTrainedModel
from typing import Sequence, List, Union, Dict, Any, Optional, Set
import numpy as np
from torch import Tensor
from transformers import Conversation
import uuid
import logging
from transformers import pipeline

class Sentence(gym.Space):
    def __init__(
        self, max_length: int, min_length: int = 1, vocabulary: Sequence[str] = None
    ):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        # TODO: The length of the sentence should vary.
        self.vocabulary: Set[str] = set(vocabulary or english_words_set)
        self.vocab_np = np.array(sorted(self.vocabulary))
        self.vocab_length = len(self.vocabulary)

    def contains(self, sample: Union[str, Any]) -> bool:
        if not isinstance(sample, str):
            return False
        if not (self.min_length <= len(sample) <= self.max_length):
            return False
        words_not_in_vocab = [
            word for word in sample.split() if word not in self.vocabulary
        ]
        if words_not_in_vocab:
            print(f"sample has words not in vocabulary: {words_not_in_vocab}")
            return False
        return True

    def sample(self):
        # TODO: Use a language model instead for sampling.
        sentence_length = self.np_random.randint(self.min_length, self.max_length + 1)
        ids = self.np_random.choice(
            self.vocab_length, size=sentence_length, replace=True
        )
        words_np = self.vocab_np[ids]
        sentence_str = " ".join(words_np)
        return sentence_str


class CustomConversation(Conversation):
    """ Optional: Extend `Conversation` so the names for the 'user' and 'bot' can be changed. """

    def __init__(
        self,
        text: str = None,
        conversation_id: uuid.UUID = None,
        past_user_inputs: List[str] = None,
        generated_responses: List[str] = None,
        user_name: str = "user",
        bot_name: str = "bot",
    ):
        super().__init__(
            text=text,
            conversation_id=conversation_id,
            past_user_inputs=past_user_inputs,
            generated_responses=generated_responses,
        )
        self.user_name = user_name
        self.bot_name = bot_name

    def __repr__(self):
        """
        Generates a string representation of the conversation.

        Return:
            :obj:`str`:

            Example: Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user >> Going to the movies tonight - any
            suggestions? bot >> The Big Lebowski
        """
        lines = [f"Conversation id: {self.uuid}"]
        for is_user, text in self.iter_texts():
            name = self.user_name if is_user else self.bot_name
            lines.append(f"{name}: {text}")
        return "\n".join(lines)

    @classmethod
    def wrap(
        cls, convo: Conversation, user_name: str = "Agent", bot_name: str = "User"
    ) -> "CustomConversation":
        return CustomConversation(
            text=convo.new_user_input,
            conversation_id=convo.uuid,
            past_user_inputs=convo.past_user_inputs,
            generated_responses=convo.generated_responses,
            user_name=user_name,
            bot_name=bot_name,
        )