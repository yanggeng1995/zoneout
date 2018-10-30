import tensorflow as tf
import numpy as np

class ZoneoutWrapper(tf.contrib.rnn.RNNCell):

    def __init__(self, cell, zoneout_prob, is_training=True):

        """
        Args:
          cell: RNNCell
          zoneout_prob: cell and state zoneout prob
          is_training: training or predicting 
        """

        super(ZoneoutWrapper, self).__init__()
        self.cell = cell
        self.zoneout_prob = zoneout_prob
        self.is_training = is_training

        if not isinstance(cell, tf.contrib.rnn.RNNCell):
            raise TypeError("The cell is not RNNCell!")
        if isinstance(self.state_size, tf.contrib.rnn.LSTMStateTuple):
            if not isinstance(zoneout_prob, tuple):
                raise TypeError("The zoneout_prob must be a tuple!")
            if len(self.state_size) != len(self.zoneout_prob):
                raise TypeError("State and zoneout need equally many parts")

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size
    
    def call(self, inputs, state, scope=None):

        output, new_state = self.cell(inputs, state, scope)
        if isinstance(self.state_size, tuple):
            if self.is_training:
                c, h = state
                new_c, new_h = new_state
                zoneout_prob_c, zoneout_prob_h = self.zoneout_prob
                new_c = (1 - zoneout_prob_c) * tf.nn.dropout(
                    new_c - c, (1 - zoneout_prob_c)) + c
                new_h = (1 - zoneout_prob_h) * tf.nn.dropout(
                    new_h - h, (1 - zoneout_prob_h)) + h
                new_state = tf.contrib.rnn.LSTMStateTuple(c=new_c, h=new_h)
                output = new_h
            else:
                c, h = state
                new_c, new_h = new_state
                zoneout_prob_c, zoneout_prob_h = self.zoneout_prob
                new_c = zoneout_prob_c * c + (1 - zoneout_prob_c) * new_c
                new_h = zoneout_prob_h * h + (1 - zoneout_prob_h) * new_h
                new_state = tf.contrib.rnn.LSTMStateTuple(c=new_c, h=new_h)
                output = new_h
        else:
            if self.is_training:
                new_state = (1 - self.zoneout_prob) * tf.nn.dropout(
                    new_state - state, (1 - self.zoneout_prob)) + state
            else:
                new_state = self.zoneout_prob * state + (1 - self.zoneout_prob) * new_state
        
        return output, new_state

if __name__ == "__main__":

    cell = ZoneoutWrapper(tf.nn.rnn_cell.LSTMCell(5), zoneout_prob=(0.1, 0.2))
    X = np.random.randn(2, 3, 4).astype(np.float32)
    X[1, 2 : ] = 0
    X_lengths = [3, 2]

    outputs, states = tf.nn.dynamic_rnn(
                        cell,
                        X,
                        sequence_length=X_lengths,
                        dtype=tf.float32,
                        scope="zoneout")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output, state = sess.run([outputs, states])
        print(output.shape)
        print(output)
        print(state[0])
        print(state[1])
