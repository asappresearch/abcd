import gym
from typing import Tuple, Any
from typing import Any, Dict
from transformers import Conversation
import logging
from transformers import pipeline
from abcd.utils.sentence import Sentence, CustomConversation
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)


class DialogueEnv(gym.Env):
    """ Gym environment that represents a conversation with a user (a LLM). """
    def __init__(self):
        super().__init__()
        # NOTE: I was using these models before, until I found out about this
        # `conversational` pipeline, which simplifies the code a lot.
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        # self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.user_pipeline = pipeline("conversational")
        # TODO: Figure out the vocabulary of ABCD / or of the tokenizer itself.
        self.observation_space = Sentence(max_length=128)
        self.action_space = Sentence(max_length=10)
        self._conversation: CustomConversation

    def reset(
        self,
        agent_prompt: str = "Hello, How can I help you?",
        user_prompt: str = None,
    ):
        """ Start a new conversation, primed with the agent say `agent_prompt`, and the
        user responding `user_prompt`.
        When `user_prompt` is not given, it is generated from the language model.
        """
        # TODO: Retrieve some kind of prompt to prime the LM from the ABCD dataset.
        # prompt = get_prompt_from_ABCD()

        # "Prime" the conversation with some dialogue, and mark it as completed.
        # This seems to help the model a little bit.
        self.conversation = Conversation(agent_prompt)
        if user_prompt:
            self.conversation.append_response(user_prompt)
            self.conversation.mark_processed()
        else:
            # Generate the first user response.
            self.conversation = self.user_pipeline(self.conversation)

        first_user_response = self.conversation.generated_responses[-1]
        return first_user_response

    def step(self, action: str) -> Tuple[str, Any, bool, Dict]:
        self.chat_history.append(action)
        self.conversation.add_user_input(action)
        # TODO: This `max_length=10_000` argument is random, but it seems to help it
        # not repeat itself.
        self.conversation = self.user_pipeline(self.conversation, max_length=10000)

        user_response: str = self.conversation.generated_responses[-1]
        
        # Assign a reward based on the user's response:
        reward: float = 0.0
        if any(v in user_response for v in ["Thanks", "Great!", "That works!"]):
            reward += 1
        if any(
            v in user_response
            for v in [
                "I'm not sure what you mean",
                "I don't understand",
                "That's not helpful",
            ]
        ):
            reward -= 1

        # End the conversation if:
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
