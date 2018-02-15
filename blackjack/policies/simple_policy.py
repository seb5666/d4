from collections import namedtuple
from experiments.nam_seq2seq import NAMSeq2Seq

import tensorflow as tf
import numpy as np

from blackjack.policies.policy import Policy

SUMMARY_LOG_DIR = "../tmp/simple_policy/summaries"


# choose add - learning rate 0.05
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")

tf.app.flags.DEFINE_integer("train_num_steps", -1, "Training phase - number of steps")
tf.app.flags.DEFINE_integer("train_stack_size", -1, "Training phase - stack size")

tf.app.flags.DEFINE_integer("test_num_steps", -1, "Testing phase - number of steps")
tf.app.flags.DEFINE_integer("test_stack_size", -1, "Testing phase - stack size")

tf.app.flags.DEFINE_integer("min_return_width", 5, "Minimum return width")

tf.app.flags.DEFINE_integer("eval_every", 5, "Evaluate every n-th step")

tf.app.flags.DEFINE_integer("max_epochs", 1000, "Maximum number of epochs")

tf.app.flags.DEFINE_string("id", "x", "unique id for summary purposes")

tf.app.flags.DEFINE_float("init_weight_stddev", 0.1, "Standard deviation for initial weights")

tf.app.flags.DEFINE_float("max_grad_norm", 1.0, "Clip gradients to this norm.")

tf.app.flags.DEFINE_float("grad_noise_eta", 0.01, "Gradient noise scale.")

tf.app.flags.DEFINE_float("grad_noise_gamma", 0.55, "Gradient noise gamma.")

tf.app.flags.DEFINE_string("dataset",
                           "./data/add/train_test_len/train4_test8/",
                           "unique id for summary purposes")

tf.app.flags.DEFINE_string("sketch", "./experiments/add/sketch_manipulate.d4", "sketch")

tf.app.flags.DEFINE_boolean("save_summary", True, "Save summary files.")


def print_flags(flags):
    print("Flag values")
    for k, v in flags.__dict__['__flags'].items():
        print('  ', k, ':', v)


FLAGS = tf.app.flags.FLAGS


class SimplePolicy(Policy):

    def __init__(self, sketch_path, sess, debug=False, num_actions =2):
        self.sketch = self._load_scaffold_from_file(sketch_path)
        self.sess = sess
        self.debug = debug
        self.num_actions  = num_actions
        d4InitParams = namedtuple(
            "d4InitParams", "stack_size value_size batch_size min_return_width init_weight_stddev")

        TrainParams = namedtuple(
            "TrainParams", "train learning_rate num_steps max_grad_norm grad_noise_eta grad_noise_gamma")

        TestParams = namedtuple("TestParams", "stack_size num_steps")

        # define the size of the stack
        # one-hot encoding for the sum, so anything from 0 to 21
        value_size = 22

        # each input is just the current sum of the player
        train_seq_length = 1
        test_seq_length = 1

        # Magic numbers taken from add.py TODO: figure out why
        stack_size = 2

        # only need a single action returned so either "0" or "1"
        min_return_width = 1

        # for now only do one batch at a time, i.e. only one environment simultaneously?
        batch_size = 1

        # Stddev for the initialization of the slot weights
        # Note: A normal distribution is used to initialize the weights.
        init_weight_stddev = 0.1

        self.d4_params = d4InitParams(stack_size=stack_size,
                                 value_size=value_size,
                                 batch_size=batch_size,
                                 min_return_width=min_return_width,
                                 init_weight_stddev=init_weight_stddev)

        train = True
        learning_rate = 0.001
        num_steps = 2  # More magic numbers... (really because there are only 2 operations to perform every time the machine is run..)
        max_grad_norm = 1.0  # Clip gradients to this norm
        grad_noise_eta = 0.01  # Gradient noise scale
        grad_noise_gamma = 0.55  # Gradient noise gamma

        self.train_params = TrainParams(train=train,
                                        learning_rate=learning_rate,
                                        num_steps=num_steps,
                                        max_grad_norm=max_grad_norm,
                                        grad_noise_eta=grad_noise_eta,
                                        grad_noise_gamma=grad_noise_gamma)

        self.test_params = TestParams(num_steps=num_steps,
                                      stack_size=stack_size)

        # Ensure that a sigmoid is used for binary operations such as >,..
        self.temperature = 1.0  # Temperature for binary operation evaluation

        self.model = NAMSeq2Seq(self.sketch,
                                self.d4_params,
                                self.train_params,
                                self.test_params,
                                debug=self.debug,
                                adjust_min_return_width=True,
                                argmax_pointers=True,
                                argmax_stacks=True,
                                temperature=self.temperature
                                )

        print("building graph...")
        self.model.build_graph()

        self.initialized = False

    def __call__(self, state, **kwargs):
        player_sum = state[0]

        input_seq = [player_sum]
        if self.debug:
            print("running model on input sequence {}...".format(input_seq))

        # TODO this should not be an argmaxed result!
        probabilities = self.model.evaluate_2(sess=self.sess, input_seq=input_seq, num_actions=self.num_actions)
        sample_action = np.argmax(np.random.multinomial(1, probabilities))
        if self.debug:
            print("Probabilities: {}".format(probabilities))
            print("Sampled action: {}".format(sample_action))
        return sample_action

    # Perform a single step of optimisation in the REINFORCE algorithm
    def update_policy(self, state, action, total_reward):
        player_sum = state[0]
        input_seq = [player_sum]
        self.model.update_policy(self.sess, input_seq, action, total_reward)

    @staticmethod
    def _load_scaffold_from_file(filename):
        with open(filename, "r") as f:
            scaffold = f.read()
        return scaffold


if __name__ == "__main__":

    with tf.Session() as sess:
        policy = SimplePolicy('sketches/simple_policy.d4', sess=sess)

        sess.run(tf.global_variables_initializer())

        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        variables = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        print(variables)

        input_seq = [14]
        results = []
        for i in range(1):
            print(i)
            r = policy.model.evaluate_2(sess=sess, input_seq=input_seq)
            results.append(r)
            print()

        variables = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        print(variables)
        #
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.figure()
        # plt.plot(np.arange(len(results)), results)
        # plt.show()