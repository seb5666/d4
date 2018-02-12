from collections import namedtuple

import numpy as np
import tensorflow as tf

from experiments.nam_seq2seq import NAMSeq2Seq
from experiments.data import load_data, DatasetBatcher


np.set_printoptions(linewidth=20000, precision=2, suppress=True, threshold=np.nan)

SUMMARY_LOG_DIR = "./tmp/add/summaries"

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
                           "../data/add/train_test_len/train4_test8/",
                           "unique id for summary purposes")

tf.app.flags.DEFINE_string("sketch", "./add/sketch_manipulate.d4", "sketch")

tf.app.flags.DEFINE_boolean("save_summary", True, "Save summary files.")


def print_flags(flags):
    print("Flag values")
    for k, v in flags.__dict__['__flags'].items():
        print('  ', k, ':', v)


FLAGS = tf.app.flags.FLAGS


d4InitParams = namedtuple(
    "d4InitParams", "stack_size value_size batch_size min_return_width init_weight_stddev")

TrainParams = namedtuple(
    "TrainParams", "train learning_rate num_steps max_grad_norm grad_noise_eta grad_noise_gamma")

TestParams = namedtuple("TestParams", "stack_size num_steps")


dataset_path = FLAGS.dataset
print('Dataset path:', dataset_path)

datasets = load_data(dataset_path)


# calculate value_size automatically
value_size = max(datasets.train.input_seq.max(), datasets.train.target_seq.max(),
                 datasets.dev.input_seq.max(), datasets.dev.target_seq.max(),
                 datasets.test.input_seq.max(), datasets.test.target_seq.max(),
                 datasets.debug.input_seq.max(), datasets.debug.target_seq.max()) + 2

print('value_size', value_size)

dataset_train = datasets.train
dataset_dev = datasets.dev
dataset_test = datasets.test

train_batcher = DatasetBatcher(dataset_train, FLAGS.batch_size)

train_seq_len = dataset_train.input_seq[:, -1].max()
test_seq_len = dataset_test.input_seq[:, -1].max()
dev_seq_len = dataset_dev.input_seq[:, -1].max()

train_num_steps = train_seq_len * 8 + 6
test_num_steps = test_seq_len * 8 + 6
dev_num_steps = dev_seq_len * 8 + 6

train_stack_size = train_seq_len * 3 + 10
test_stack_size = test_seq_len * 3 + 10

FLAGS.train_num_steps = (train_num_steps if FLAGS.train_num_steps == -1
                         else FLAGS.train_num_steps)
FLAGS.train_stack_size = (train_stack_size if FLAGS.train_stack_size == -1
                          else FLAGS.train_stack_size)

FLAGS.test_num_steps = (test_num_steps if FLAGS.test_num_steps == -1
                        else FLAGS.test_num_steps)
FLAGS.test_stack_size = (test_stack_size if FLAGS.test_stack_size == -1
                         else FLAGS.test_stack_size)


d4_params = d4InitParams(stack_size=FLAGS.train_stack_size,
                         value_size=value_size,
                         batch_size=FLAGS.batch_size,
                         min_return_width=FLAGS.min_return_width,
                         init_weight_stddev=FLAGS.init_weight_stddev
                         )

train_params = TrainParams(train=True,
                           learning_rate=FLAGS.learning_rate,
                           num_steps=FLAGS.train_num_steps,
                           max_grad_norm=FLAGS.max_grad_norm,
                           grad_noise_eta=FLAGS.grad_noise_eta,
                           grad_noise_gamma=FLAGS.grad_noise_gamma
                           )

test_params = TestParams(num_steps=FLAGS.test_num_steps,
                         stack_size=FLAGS.test_stack_size
                         )



load_dir = "../models"
print("Save dir: {}".format(load_dir))

def load_scaffold_from_file(filename):
    with open(filename, "r") as f:
        scaffold = f.read()
    return scaffold

sketch = load_scaffold_from_file(FLAGS.sketch)


model = NAMSeq2Seq(sketch, d4_params, train_params, test_params,
                   debug=True,
                   adjust_min_return_width=True,
                   argmax_pointers=True,
                   argmax_stacks=True,
                   )

with tf.Session() as sess:
    model.build_graph()

    model.load_model(sess, load_dir)

    # model.run_eval_step(sess=sess, dataset=dataset_test, max_steps=100)

    input_seq = [8,1,1,1,0,9,0,3]   # 810 + 119 + (0 carry)  ---- (len 3)
    target_seq = [1,0,9,2,9]            # = 0929
    result = model.evaluate(sess=sess, input_seq=input_seq, target_seq=target_seq, max_steps=100)
    print("Result: {}".format(result))
