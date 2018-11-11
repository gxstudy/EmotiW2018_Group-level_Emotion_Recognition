import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

np.set_printoptions(threshold=np.nan)


test = np.load('../data/test_inorder.npy')
print test.shape
num_test_sample = test.shape[0]
print num_test_sample
 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# hyper-parameters
learning_rate = 0.001
batch_size = tf.placeholder(tf.int32)
input_size = 2048
timestep_size = 16
hidden_size = 128
layer_num = 1
class_num = 3


X=tf.placeholder("float",[None,timestep_size,input_size])
Y=tf.placeholder("float",[None,class_num])


keep_prob = tf.placeholder(tf.float32)


def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(hidden_size,
                                        forget_bias=1.0,
                                        state_is_tuple=True)#resue = tf.get_variable_scope().reuse

def attn_cell():
    return tf.contrib.rnn.DropoutWrapper(lstm_cell(),
                                         output_keep_prob=keep_prob)

mlstm_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(layer_num)])

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

outputs = list()
state = init_state
with tf.variable_scope("RNN"):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
h_state = outputs[-1]


# loss & optimizer
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)


sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

with tf.Session() as sess:  
    saver.restore(sess, "./modes/attention.ckpt")
    test_predicted_probs=sess.run(y_pre,feed_dict={X: test,                                             
                                              keep_prob: 1.0,
                                              batch_size: num_test_sample})

a = np.argwhere(np.isnan(test_predicted_probs))
np.save('test_attention_probs', test_predicted_probs)
