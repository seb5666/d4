
import tensorflow as tf
import numpy as np

from blackjack.blackjack_machine import BlackjackMachine
from blackjack.policies.policy import Policy


class SimplePolicy(Policy):

    def __init__(self, sketch_path, sess, debug=False, num_actions =2):
        self.sketch = self._load_scaffold_from_file(sketch_path)
        self.sess = sess
        self.debug = debug
        self.num_actions = num_actions

        self.blackjack_machine = BlackjackMachine(self.sketch)

        self.initialized = False

    def __call__(self, state, **kwargs):
        player_sum = state[0]
        input_seq = [player_sum]

        if self.debug:
            print("running model on input sequence {}...".format(input_seq))

        probabilities = self.blackjack_machine.run_train_step(sess=self.sess, input_seq=input_seq)
        sample_action = np.argmax(np.random.multinomial(1, probabilities))
        if self.debug:
            print("Probabilities: {}".format(probabilities))
            print("Sampled action: {}".format(sample_action))
        return sample_action

    # Perform a single step of optimisation in the REINFORCE algorithm
    def update_policy(self, state, action, total_reward):
        player_sum = state[0]
        input_seq = [player_sum]
        self.blackjack_machine.run_update_step(self.sess, input_seq, action, total_reward)

    @staticmethod
    def _load_scaffold_from_file(filename):
        with open(filename, "r") as f:
            scaffold = f.read()
        return scaffold


if __name__ == "__main__":

    with tf.Session() as sess:
        policy = SimplePolicy2('sketches/simple_policy.d4', sess=sess)

        sess.run(tf.global_variables_initializer())

        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        variables = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        print(variables)

        input_seq = [14]
        results = []
        for i in range(1):
            print(i)
            r = policy.blackjack_machine.run_train_step(sess=sess, input_seq=input_seq)
            results.append(r)
            print(r)

        variables = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        print(variables)
