from d4.dsm.loss import ReinforceLoss
from d4.interpreter import SimpleInterpreter

import tensorflow as tf
import numpy as np

class BlackjackMachine:

    def __init__(self,
                 sketch):

        self.sketch = sketch

        # TODO: get params as arguments
        self.stack_size = 2
        self.value_size = 22
        self.num_steps = 2
        self.min_return_width = 1
        self.batch_size = 1  # Note: Has to be 1 for now
        self.init_weight_stddev = 0.1
        self.temperature = 1.0
        self.learning_rate = 0.01

        self.interpreter = SimpleInterpreter(stack_size=self.stack_size,
                                             value_size=self.value_size,
                                             min_return_width=self.min_return_width,
                                             batch_size=self.batch_size,
                                             init_weight_stddev=self.init_weight_stddev,
                                             temperature=self.temperature
                                             )

        for batch in range(self.batch_size):
            self.interpreter.load_code(self.sketch, batch)

        self.interpreter.create_initial_dsm()

        trace = self.interpreter.execute(self.num_steps)
        self._trace = trace

        self.reinforce_loss = ReinforceLoss(trace[-1], self.interpreter)

        # TODO: experiment with initial learning rate
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # We want to maximise the loss, hence multiply by -1
        self.train_op = optimizer.minimize(-1 * self.reinforce_loss.loss)

    def run_update_step(self, sess, input_seq, action, total_reward):
        # TODO: handle batch_size greater than 1
        self.interpreter.load_stack(input_seq, 0)
        self.reinforce_loss.load_action_and_rewards([action], [total_reward])

        feed_in = self.reinforce_loss.current_feed_dict()
        sess.run(self.train_op, feed_in)

    def run_train_step(self, sess, input_seq, debug=False):

        # TODO: handle multiple batches...
        # # feed the stack and the target stack
        # for j in range(self.batch_size):
        #     self.interpreter.load_stack(input_seq[j], j, last_float=False)

        self.interpreter.load_stack(input_seq, 0)

        feed_in = self.reinforce_loss.current_feed_dict()

        traces = self.interpreter.execute(self.num_steps)
        final_dsm = traces[-1]
        final_stack = final_dsm.data_stack

        final_stack_ = sess.run(final_stack, feed_in)

        probabilities = final_stack_[:2, 0, 0]  # Only 2 actions (hit/stick)
        return probabilities

    def run_eval_step(self, sess, input_seq, debug=False):

        self.interpreter.test_time_load_stack(input_seq, 0)

        #run da thing
        test_trace, _ = self.interpreter.execute_test_time(sess,
                                                           self.num_steps,
                                                           use_argmax_pointers=True,
                                                           use_argmax_stacks=True,
                                                           debug=debug,
                                                           save_only_last_step=True)

        # pull out stacks
        final_state = test_trace[-1]
        data_stacks = final_state[self.interpreter.test_time_data_stack]
        data_stack_pointers = final_state[self.interpreter.test_time_data_stack_pointer]

        # argmax everything !!
        # print("data_stack_pointer: {}".format(data_stack_pointers))
        pointer = np.argmax(data_stack_pointers[:, 0])

        probabilities = data_stacks[:, 0:pointer + 1, 0].squeeze()
        result = np.argmax(probabilities, 0)
        return result