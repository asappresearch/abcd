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
from abcd.utils.sentence import Sentence, CustomConversation
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)


class DialogueEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        # self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        # self.text_generator = pipeline("text-generation")
        self.user_pipeline = pipeline("conversational")
        # TODO: Figure out the vocabulary of ABCD / or of the tokenizer itself.
        # tokenizer_vocab = self.tokenizer.get_vocab()
        self.observation_space = Sentence(max_length=128)
        self.action_space = Sentence(max_length=10)
        # self.observation_space = Word(max_length=128, vocab=english_words_set)
        # self.action_space = Word(max_length=20, vocab=english_words_set)
        self._conversation: CustomConversation
        self.conversation = Conversation()
        self.prompt: str
        self.prompt_ids: Tensor
        self.chat_history: List[str] = []
        self.chat_history_ids: Tensor

    def reset(
        self,
        agent_prompt: str = "Hello, How can I help you?",
        user_prompt: str = "I'm having trouble with my laptop.",
    ):
        # TODO: Retrieve some kind of prompt to prime the LM from the ABCD dataset.
        # prompt = get_prompt_from_ABCD()

        # "Prime" the conversation with some dialogue, and mark it as completed.
        # This seems to help the model a little bit.
        self.conversation = Conversation(agent_prompt)
        if user_prompt:
            self.conversation.append_response(user_prompt)
            self.conversation.mark_processed()
        else:
            self.conversation = self.user_pipeline(self.conversation)

        first_user_response = self.conversation.generated_responses[-1]
        return first_user_response

    def step(self, action: str) -> Tuple[str, Any, bool, Dict]:
        self.chat_history.append(action)
        self.conversation.add_user_input(action)
        # TODO: At some point the language model keeps repeating the same thing over an
        # over, not sure why exactly. Maybe it's always trying to recreate the whole
        # conversation, and that's longer than the max_length of the generative model?
        # IDEA: Truncate the conversation, hopefully that will help:
        # self.conversation.past_user_inputs = self.conversation.past_user_inputs[-5:]
        # self.conversation.generated_responses = self.conversation.generated_responses[-5:]

        self.conversation = self.user_pipeline(self.conversation, max_length=10000)

        user_response: str = self.conversation.generated_responses[-1]
        reward: float = 0.0
        if any(v in user_response for v in ["Thanks", "Great!", "That works!"]):
            reward += 1
        if any(
            v in user_response
            for v in [
                "I'm not sure what you mean",
                "I don't understand",
                "That's not helpful.",
            ]
        ):
            reward -= 1

        # Kill the conversation if:
        # - the user doesn't respond
        # - the conversation ends peacefully
        # - the conversation derails too much.
        user_response_lowercase = user_response.lower()
        done = not user_response or any(
            v.lower() in user_response_lowercase
            for v in [
                "Good bye",
                "Goodbye",
                "You too!",
                "Good night",
                "have a good day",
            ]
        )
        return user_response, reward, done, {}

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @conversation.setter
    def conversation(self, value: Conversation):
        self._conversation = CustomConversation.wrap(
            value, user_name="Agent", bot_name="User"
        )


def main():
    env = DialogueEnv()
    obs = env.reset()
    print(env.conversation)

    done = False
    steps = 0
    while not done:
        # Option 1: Get custom input from the 'agent':
        try:
            action = input(f"Agent: ")
        except KeyboardInterrupt:
            break
        # Option 2: Say random words to the user (not very helpful!)
        # action = env.action_space.sample()
        # print(f"Agent: {action}")

        obs, reward, done, info = env.step(action)
        print(f"User: {obs} \t (Reward: {reward}, done: {done}, info: {info})")
        steps += 1

        if steps > 10:
            print(f"Exiting since we reached {10} steps.")
            break

if __name__ == "__main__":
    main()
