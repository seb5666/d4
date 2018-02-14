from collections import namedtuple
from experiments.nam_seq2seq import NAMSeq2Seq
import os
import tensorflow as tf
from .policy import Policy

SUMMARY_LOG_DIR = "../tmp/simple_policy/summaries"

class SimplePolicy(Policy):

    def __init__(self, sketch_path, debug=False):
        self.sketch = self._load_scaffold_from_file(sketch_path)
        self.debug = debug
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
        stack_size = 3

        # only need a single action returned so either "0" or "1"
        min_return_width = 1

        # for now only do one batch at a time, i.e. only one environment simultaneously?
        batch_size = 1

        # Stddev for the initialization of the slot weights
        # Note: A normal distribution is used to initialize the weights.
        init_weight_stddev = 1.0

        self.d4_params = d4InitParams(stack_size=stack_size,
                                 value_size=value_size,
                                 batch_size=batch_size,
                                 min_return_width=min_return_width,
                                 init_weight_stddev=init_weight_stddev)

        train = True
        learning_rate = 0.01
        num_steps = 3  # More magic numbers...
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

        self.model = NAMSeq2Seq(self.sketch,
                                self.d4_params,
                                self.train_params,
                                self.test_params,
                                debug=self.debug,
                                adjust_min_return_width=True,
                                argmax_pointers=True,
                                argmax_stacks=True,
                                )

        print("building graph...")
        self.model.build_graph()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR + "/" + "simple_policy",
                                                   tf.get_default_graph())
            print(tf.GraphKeys.TRAINABLE_VARIABLES)
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                print(i) # i.name if you want just a name
            print("Initializing global tf variables...")
            sess.run(tf.global_variables_initializer())

    def __call__(self, state, **kwargs):
        player_sum = state[0]

        with tf.Session() as sess:

            input_seq = [player_sum]
            if self.debug:
                print("running model on input sequence {}...".format(input_seq))

            result = self.model.evaluate(sess=sess, input_seq=input_seq, max_steps=self.train_params.num_steps)
            if self.debug:
                print("Result: {}".format(result))

            action = result[-1]
            return action

    @staticmethod
    def _load_scaffold_from_file(filename):
        with open(filename, "r") as f:
            scaffold = f.read()
        return scaffold

