import tensorflow as tf
import numpy as np

class Rev_gen_pre(object):
    def __init__(self, rev_num_u, rev_num_i, rev_len_u, rev_len_i, rev_len, user_num, item_num, user_vocab_size,
                 item_vocab_size, n_latent, word_emb_size, filter_sizes, num_filters_his,
                 batch_size, rnn_size):
        self.input_u = tf.placeholder(tf.int32, [batch_size, rev_num_u, rev_len_u], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [batch_size, rev_num_i, rev_len_i], name="input_i")
        self.input_reuid = tf.placeholder(tf.int32, [None, rev_num_u], name='input_reuid')
        self.input_reiid = tf.placeholder(tf.int32, [None, rev_num_i], name='input_reuid')
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None], name="input_iid")
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.iidW = tf.Variable(tf.random_uniform([item_num + 2, n_latent], -0.1, 0.1), name="iidW")
        self.uidW = tf.Variable(tf.random_uniform([user_num + 2, n_latent], -0.1, 0.1), name="uidW")
        self.input_rev = tf.placeholder(tf.int32, [None, rev_len])

        self.l2_loss = tf.constant(0.0)
        with tf.name_scope("u_i_his_embedding"):   #users/items history reviews emb
            self.U_vocab = tf.Variable(tf.random_uniform([user_vocab_size, word_emb_size], -1.0, 1.0), name="U_vocab")
            self.emb_u_his = tf.nn.embedding_lookup(self.U_vocab, self.input_u)
            self.I_vocab = tf.Variable(tf.random_uniform([item_vocab_size, word_emb_size], -1.0, 1.0), name="I_vocab")
            self.emb_i_his = tf.nn.embedding_lookup(self.I_vocab, self.input_i)

            self.emb_u_his = tf.reshape(self.emb_u_his, [-1, rev_len_u, word_emb_size])   #rnn
            self.emb_i_his = tf.reshape(self.emb_i_his, [-1, rev_len_i, word_emb_size])
            self.u_his_revs = self.word_BiRNN_u_his(self.emb_u_his, batch_size*rev_num_u, word_emb_size, rnn_size, rev_len_u, self.dropout) #(-1, rnn_size)
            self.i_his_revs = self.word_BiRNN_i_his(self.emb_i_his, batch_size*rev_num_i, word_emb_size, rnn_size, rev_len_i, self.dropout)
            self.u_his_revs = tf.reshape(self.u_his_revs, [batch_size, rev_num_u, rnn_size])
            self.i_his_revs = tf.reshape(self.i_his_revs, [batch_size, rev_num_i, rnn_size])

            self.u_feas_his = tf.reduce_mean(self.u_his_revs, axis=1)    #[batch_size, rnn_size]
            self.i_feas_his = tf.reduce_mean(self.i_his_revs, axis=1)

            self.u_feas_his_drop = tf.nn.dropout(self.u_feas_his, self.dropout)
            self.i_feas_his_drop = tf.nn.dropout(self.i_feas_his, self.dropout)


        with tf.name_scope("feas_cross"):
            self.user_emb = self.u_feas_his_drop
            self.item_emb = self.i_feas_his_drop


        with tf.name_scope("pre_loss"):
            self.score = tf.reduce_sum(tf.multiply(self.user_emb, self.item_emb), axis=1)
            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.bised = tf.Variable(tf.constant(0.1), name='bias')
            self.predictions = self.score + self.u_bias + self.i_bias + self.bised

            self.loss = tf.reduce_sum(tf.square(self.predictions - self.input_y)) + 0.1 * tf.trace(
                tf.matmul(tf.transpose(self.user_emb), self.user_emb)) \
                        + 0.1 * tf.trace(tf.matmul(tf.transpose(self.item_emb), self.item_emb)) + \
                        0.1 * tf.reduce_sum(tf.square(self.u_bias)) + 0.1 * tf.reduce_sum(tf.square(self.i_bias))

        with tf.name_scope("eval"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.mse = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))

    def word_CNN(self, x, batch_size, word_emb_size, rnn_size, rev_len, dropout):
        pooled_outputs = []
        filter_sizes = [3]
        x = tf.expand_dims(x, -1)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, word_emb_size, 1, rnn_size]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[rnn_size]), name="b")
                conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, rev_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = rnn_size * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        h_drop = tf.nn.dropout(h_pool_flat, dropout)
        return h_drop

    def word_BiRNN_u_his(self, x, batch_size, word_emb_size, rnn_size, rev_len, dropout):
        with tf.name_scope("bi_rnn_u"):
            x = tf.reshape(x, [-1, word_emb_size])
            W_in = tf.Variable(tf.random_normal([word_emb_size, rnn_size], mean=0.0, stddev=np.sqrt(2. / (word_emb_size + rnn_size))), name='W_in')
            biases_in = tf.Variable(tf.constant(0.1, shape=[rnn_size]), dtype=tf.float32)
            x_in = tf.matmul(x, W_in) + biases_in
            x_in_drop = tf.nn.dropout(x_in, dropout)
            x_in_drop = tf.reshape(x_in_drop, [-1, rev_len, rnn_size])
            sequence_len = np.array([rev_len] * batch_size)
            with tf.variable_scope('init_name_u', initializer=tf.orthogonal_initializer()):  # 正交初始化
                LSTM_cell_fw = tf.contrib.rnn.BasicLSTMCell(int(rnn_size / 2))
                LSTM_cell_bw = tf.contrib.rnn.BasicLSTMCell(int(rnn_size / 2))
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(LSTM_cell_fw, LSTM_cell_bw, x_in_drop, sequence_len,
                                                                    dtype=tf.float32)
            bi_rnn_out = tf.concat(outputs, 2)[:, -1, :]

        return bi_rnn_out

    def word_BiRNN_i_his(self, x, batch_size, word_emb_size, rnn_size, rev_len, dropout):
        with tf.name_scope("bi_rnn_i"):
            x = tf.reshape(x, [-1, word_emb_size])
            W_in = tf.Variable(tf.random_normal([word_emb_size, rnn_size], mean=0.0, stddev=np.sqrt(2. / (word_emb_size + rnn_size))), name='W_in')
            biases_in = tf.Variable(tf.constant(0.1, shape=[rnn_size]), dtype=tf.float32)
            x_in = tf.matmul(x, W_in) + biases_in
            x_in_drop = tf.nn.dropout(x_in, dropout)
            x_in_drop = tf.reshape(x_in_drop, [-1, rev_len, rnn_size])
            sequence_len = np.array([rev_len] * batch_size)
            with tf.variable_scope('init_name_i', initializer=tf.orthogonal_initializer()):  # 正交初始化
                GRU_cell_fw = tf.contrib.rnn.GRUCell(int(rnn_size / 2))
                GRU_cell_bw = tf.contrib.rnn.GRUCell(int(rnn_size / 2))
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(GRU_cell_fw, GRU_cell_bw, x_in_drop, sequence_len,
                                                                    dtype=tf.float32)
            bi_rnn_out = tf.concat(outputs, 2)[:, -1, :]
        return bi_rnn_out

    def atten_u_i_revs(self, u_revs, i_revs, rnn_size, n_latent):
        # attention
        self.Wau = tf.Variable(tf.random_uniform([rnn_size, n_latent], -0.1, 0.1), name='Wau')
        self.Wru = tf.Variable(tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wru')
        self.Wpu = tf.Variable(tf.random_uniform([n_latent, 1], -0.1, 0.1), name='Wpu')
        self.bau = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bau")
        self.bbu = tf.Variable(tf.constant(0.1, shape=[1]), name="bbu")
        self.iid_a = tf.nn.relu(tf.nn.embedding_lookup(self.iidW, self.input_reuid))
        self.u_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(tf.einsum('ajk,kl->ajl', u_revs, self.Wau) +
                    tf.einsum('ajk,kl->ajl', self.iid_a,self.Wru) + self.bau),self.Wpu) + self.bbu  # None*u_len*1
        self.u_a = tf.nn.softmax(self.u_j, 1)  # none*u_len*1

        self.Wai = tf.Variable(tf.random_uniform([rnn_size, n_latent], -0.1, 0.1), name='Wai')
        self.Wri = tf.Variable(tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wri')
        self.Wpi = tf.Variable(tf.random_uniform([n_latent, 1], -0.1, 0.1), name='Wpi')
        self.bai = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bai")
        self.bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")
        self.uid_a = tf.nn.relu(tf.nn.embedding_lookup(self.uidW, self.input_reiid))
        self.i_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(tf.einsum('ajk,kl->ajl', i_revs, self.Wai) +
                        tf.einsum('ajk,kl->ajl', self.uid_a,self.Wri) + self.bai),self.Wpi) + self.bbi
        self.i_a = tf.nn.softmax(self.i_j, 1)  # none*len*1

        self.l2_loss += tf.nn.l2_loss(self.Wau)
        self.l2_loss += tf.nn.l2_loss(self.Wru)
        self.l2_loss += tf.nn.l2_loss(self.Wri)
        self.l2_loss += tf.nn.l2_loss(self.Wai)

        u_feas = tf.reduce_sum(tf.multiply(self.u_a, u_revs), 1)
        i_feas = tf.reduce_sum(tf.multiply(self.i_a, i_revs), 1)

        self.U_to_latent_W = tf.Variable(tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='U_to_latent')
        self.U_to_latent_b = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="U_to_latent_b")
        self.I_to_latent_W = tf.Variable(tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='I_to_latent_W')
        self.I_to_latent_b = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="I_to_latent_b")
        u_feas = tf.nn.tanh(tf.matmul(u_feas, self.U_to_latent_W) + self.U_to_latent_b)
        i_feas = tf.nn.tanh(tf.matmul(i_feas, self.I_to_latent_W) + self.I_to_latent_b)
        return u_feas, i_feas


