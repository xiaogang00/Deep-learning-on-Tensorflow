import time  
import numpy as np  
import tensorflow as tf   
import reader  

class PTBInput(onject):
    def __init__(self, config, data, name = None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) //num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name = name   
        )

class PTBModel(object):
    def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

del lstm_cell():
   return tf.contrib.rnn.BasicLSTMCell()(
       size, forget_bias = 0.0, state_is_tuple = True  
   )
attn_cell = lstm_cell
if is_training and config.keep_prob < 1:
    def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = config.keep_prob)
celll = tf.contrib.rnn.MultiRNNCell(
    [attm_cell()  for _ in range(config.num_layers)],
    state_is_tuple = True
)
self._initial_state = cell.zero_state(batch_size, tf.float32)

with rf.device("/cpu:0"):
    embedding = tf.get_variable(
        "embedding", [vocab_size, size], dsize = tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

if is_training and config.keep_prob < 1:
    inputs = tf.nn.dropout(inputs, config.keep_prob)


outputs = []
state = self._initial_state
with tf.variable_scope("RNN"):
    for time_step in range(num_steps):
        if time_step > 0 :
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

output  = tf.reshape(tf.concat(outputs, 1), [-1, size])
softmax_w = tf.get_variable(
    "softmax_w", [size, vocab_size], dtype = tf.float32)
softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
logits = tf.matmul(output, softmax_w) + softmax_b
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    [logits],
    [tf.reshape(input_.targets, [-1])],
    [tf.ones([batch_size * num_steps], dtype = tf.float32)])
self._cost = cost = tf.reduce_num(loss) / batch_size
self._final_state = state   

if not is_training:
    return 

self._lr = tf.Variable(0.0, trainable = False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
optimizer = tf.train.GradientDescentOptimizer(self._lr)
self._train_op = optimizer.apply_gradients(zip(grads, tvars),
global_step = tf.contrib.framework.get_or_create_global_step())

self._new_lr = tf.placeholder(
    tf.float32, shape = [], name = "new_learning_rate")
self._lr_update = tf.assign(self._lr, self._new_lr)
def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict = {self._new_lr:lr_value})


@property
def input(self):
    return self._input

@property
def _initial_state(self):
    return self._initial_state

@property
def cost(self):
    return self._cost

@property
def final_state(self):
    return self._final_state

@property
def lr(self):
    return self._lr

@property
def train_op(self):
    return self._train_op


class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def run_epoch(session, model, eval_op = None, verbose = False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state":model.final_state,
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op
    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i , (c,h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity:%.3f  speed:%.0f wps" %
            (step * 1.0/model.input.epoch_size, np.exp(costs/iters),
            iters * model.input.batch_size / (time.time() - start_time)))
    return np.exp(costs /iters)