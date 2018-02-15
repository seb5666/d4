
import tensorflow as tf
import numpy as np

from blackjack.blackjack_machine import BlackjackMachine
from blackjack.policies.policy import Policy


class SimplePolicy(Policy):

    def __init__(self, sketch_path, sess, debug=False):
        self.sketch = self._load_scaffold_from_file(sketch_path)
        self.sess = sess
        self.debug = debug

        self.blackjack_machine = BlackjackMachine(self.sketch)


    def _player_sums(self, state):
        player_sum = state[0]
        return np.array([player_sum])

    def __call__(self, state, argmax_stack = False, **kwargs):

        input_seq = self._player_sums(state)
        if self.debug:
            print("Evaluating policy on input sequence {}...".format(input_seq))

        if not(argmax_stack):

            probabilities = self.blackjack_machine.run_train_step(sess=self.sess, input_seq=input_seq)

            sample_action = np.argmax(np.random.multinomial(1, probabilities))

            if self.debug:
                print("Probabilities: {}".format(probabilities))
                print("Sampled action: {}".format(sample_action))

            return sample_action

        else:
            action = self.blackjack_machine.run_eval_step(sess=self.sess, input_seq=input_seq)
            return action


    # Perform a single step of optimisation in the REINFORCE algorithm
    def update_policy(self, state, action, total_reward):
        input_seq = self._player_sums(state)
        self.blackjack_machine.run_update_step(self.sess, input_seq, action, total_reward)

    @staticmethod
    def _load_scaffold_from_file(filename):
        with open(filename, "r") as f:
            scaffold = f.read()
        return scaffold
