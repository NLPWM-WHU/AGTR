import tensorflow as tf
import numpy as np

class Rev_gen_pre(object):
    def __init__(self, rev_num_u, rev_num_i, rev_len_u, rev_len_i, rev_len, user_num, item_num, user_vocab_size,
                 item_vocab_size, vocab_size, n_latent, word_emb_size, filter_sizes, num_filters_his,
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
            self.pad_word_emb = tf.Variable(np.zeros([1, word_emb_size], np.float32), name="pad_word_emb", trainable=False)
            self.U_vocab = tf.Variable(tf.random_uniform([user_vocab_size-1, word_emb_size], -1.0, 1.0), name="U_vocab")
            self.U_vocab = tf.concat([self.pad_word_emb, self.U_vocab], axis=0)
            self.emb_u_his = tf.nn.embedding_lookup(self.U_vocab, self.input_u)
            self.I_vocab = tf.Variable(tf.random_uniform([item_vocab_size-1, word_emb_size], -1.0, 1.0), name="I_vocab")
            self.I_vocab = tf.concat([self.pad_word_emb, self.I_vocab], axis=0)
            self.emb_i_his = tf.nn.embedding_lookup(self.I_vocab, self.input_i)


            self.emb_u_his = tf.expand_dims(self.emb_u_his, -1)    # cnn
            self.emb_i_his = tf.expand_dims(self.emb_i_his, -1)
            self.u_feas_his, self.i_feas_his= self.u_i_his_revs_emb_cnn(filter_sizes, word_emb_size,
                                            num_filters_his, rev_len_u, rev_len_i, rev_num_u, rev_num_i,n_latent)

            # self.emb_u_his = tf.reshape(self.emb_u_his, [-1, rev_len_u, word_emb_size])   #rnn
            # self.emb_i_his = tf.reshape(self.emb_i_his, [-1, rev_len_i, word_emb_size])
            # self.u_his_revs = self.word_BiRNN_u_his(self.emb_u_his, batch_size*rev_num_u, word_emb_size, rnn_size, rev_len_u, self.dropout) #(-1, rnn_size)
            # self.i_his_revs = self.word_BiRNN_i_his(self.emb_i_his, batch_size*rev_num_i, word_emb_size, rnn_size, rev_len_i, self.dropout)
            # self.u_his_revs = tf.reshape(self.u_his_revs, [batch_size, rev_num_u, rnn_size])
            # self.i_his_revs = tf.reshape(self.i_his_revs, [batch_size, rev_num_i, rnn_size])
            #
            # self.u_feas_his = tf.reduce_mean(self.u_his_revs, axis=1)    #[batch_size, rnn_size]
            # self.i_feas_his = tf.reduce_mean(self.i_his_revs, axis=1)
            #atten
            # self.u_feas_his, self.i_feas_his = self.atten_u_i_revs(self.u_his_revs, self.i_his_revs, rnn_size, n_latent)


            self.u_feas_his_drop = tf.nn.dropout(self.u_feas_his, self.dropout)
            self.i_feas_his_drop = tf.nn.dropout(self.i_feas_his, self.dropout)


        with tf.name_scope("u_i_rev_embedding"):
            self.Rev_vocab = tf.Variable(tf.random_uniform([vocab_size, word_emb_size], -0.1, 0.1), name="Rev_vocab")
            self.rev = tf.nn.embedding_lookup(self.Rev_vocab, self.input_rev)

            #self.rev = self.word_BiRNN(self.rev, batch_size, word_emb_size, rnn_size, rev_len, self.dropout)
            self.rev = self.word_CNN(self.rev, batch_size, word_emb_size, rnn_size, rev_len, self.dropout)
            self.u_feas_rev, self.i_feas_rev = self.rev_to_u_i(rnn_size, n_latent)

        with tf.name_scope("feas_cross"):
            W_left_u = tf.Variable(tf.random_normal([n_latent, n_latent], stddev=0.1), name='W_left_u')
            W_right_u = tf.Variable(tf.random_normal([n_latent, n_latent],stddev=0.1), name='W_right_u',)
            self.user_emb = (tf.nn.tanh(tf.matmul(self.u_feas_his_drop, W_left_u)) + tf.nn.tanh(tf.matmul(self.u_feas_rev, W_right_u)))

            W_left_i = tf.Variable(tf.random_normal([n_latent, n_latent], stddev=0.1), name='W_left_i')
            W_right_i = tf.Variable(tf.random_normal([n_latent, n_latent], stddev=0.1), name='W_right_i')
            self.item_emb = (tf.nn.tanh(tf.matmul(self.i_feas_his_drop, W_left_i)) + tf.nn.tanh(tf.matmul(self.i_feas_rev, W_right_i)))

            # self.alpha = tf.Variable(tf.constant(0.5, shape=[batch_size, 1]), name="alpha")
            # self.beta = tf.Variable(tf.constant(0.5, shape=[batch_size, 1]), name="beta")
            # self.user_emb = tf.multiply(self.alpha, self.u_feas_his_drop) + tf.multiply(self.beta, self.u_feas_rev)
            # self.item_emb = tf.multiply(self.alpha, self.i_feas_his_drop) + tf.multiply(self.beta, self.i_feas_rev)

            # self.alpha = tf.placeholder(tf.float32, name="dropout")
            # self.user_emb = self.alpha*self.u_feas_his_drop + (1-self.alpha)*self.u_feas_rev
            # self.item_emb = self.alpha*self.i_feas_his_drop + (1-self.alpha)*self.i_feas_rev

            # self.user_emb = self.u_feas_his_drop
            # self.item_emb = self.i_feas_his_drop


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

            self.reward_loss = tf.square(self.predictions - self.input_y)

        with tf.name_scope("eval"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.mse = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))

            self.predictions_his = tf.reduce_sum(tf.multiply(self.u_feas_his_drop, self.i_feas_his_drop), axis=1) + self.u_bias + self.i_bias + self.bised
            self.mae_his = tf.reduce_mean(tf.abs(tf.subtract(self.predictions_his, self.input_y)))
            self.mse_his = tf.reduce_mean(tf.square(tf.subtract(self.predictions_his, self.input_y)))

            self.predictions_rev = tf.reduce_sum(tf.multiply(self.u_feas_rev, self.i_feas_rev),
                                                 axis=1) + self.u_bias + self.i_bias + self.bised
            self.mae_rev = tf.reduce_mean(tf.abs(tf.subtract(self.predictions_rev, self.input_y)))
            self.mse_rev = tf.reduce_mean(tf.square(tf.subtract(self.predictions_rev, self.input_y)))

    def u_i_his_revs_emb_cnn(self, filter_size, word_emb_size, num_filters, rev_len_u, rev_len_i,rev_num_u, rev_num_i,n_latent):
        with tf.name_scope("user_conv"):
            # Convolution Layer
            filter_shape = [filter_size, word_emb_size, 1, num_filters]
            self.u_conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="u_conv_W")
            self.u_conv_b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="u_conv_b")
            self.emb_u_his = tf.reshape(self.emb_u_his, [-1, rev_len_u, word_emb_size, 1])
            conv = tf.nn.conv2d(self.emb_u_his,self.u_conv_W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, self.u_conv_b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h,ksize=[1, rev_len_u - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool")
        h_pool_u = pooled
        h_pool_flat_u = tf.reshape(h_pool_u, [-1, rev_num_u, num_filters])

        with tf.name_scope("item_conv"):
            # Convolution Layer
            filter_shape = [filter_size, word_emb_size, 1, num_filters]
            self.i_conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="i_conv_W")
            self.i_conv_b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="i_conv_b")
            self.emb_i_his = tf.reshape(self.emb_i_his, [-1, rev_len_i, word_emb_size, 1])
            conv = tf.nn.conv2d(self.emb_i_his,self.i_conv_W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, self.i_conv_b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h,ksize=[1, rev_len_i - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool")
        h_pool_i = pooled
        h_pool_flat_i = tf.reshape(h_pool_i, [-1, rev_num_i, num_filters])

        #attention
        self.Wau= tf.Variable(tf.random_uniform([num_filters, n_latent], -0.1, 0.1), name='Wau')
        self.Wru = tf.Variable(tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wru')
        self.Wpu = tf.Variable(tf.random_uniform([n_latent, 1], -0.1, 0.1), name='Wpu')
        self.bau = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bau")
        self.bbu = tf.Variable(tf.constant(0.1, shape=[1]), name="bbu")
        self.iid_a = tf.nn.relu(tf.nn.embedding_lookup(self.iidW, self.input_reuid))
        self.u_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
            tf.einsum('ajk,kl->ajl', h_pool_flat_u, self.Wau) + tf.einsum('ajk,kl->ajl', self.iid_a, self.Wru) + self.bau),
                             self.Wpu) + self.bbu  # None*u_len*1

        self.u_a = tf.nn.softmax(self.u_j, 1)  # none*u_len*1

        self.Wai = tf.Variable(tf.random_uniform([num_filters, n_latent], -0.1, 0.1), name='Wai')
        self.Wri = tf.Variable(tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wri')
        self.Wpi = tf.Variable(tf.random_uniform([n_latent, 1], -0.1, 0.1), name='Wpi')
        self.bai = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bai")
        self.bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")
        self.uid_a = tf.nn.relu(tf.nn.embedding_lookup(self.uidW, self.input_reiid))
        self.i_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
            tf.einsum('ajk,kl->ajl', h_pool_flat_i, self.Wai) + tf.einsum('ajk,kl->ajl', self.uid_a, self.Wri) + self.bai),
                             self.Wpi) + self.bbi
        self.i_a = tf.nn.softmax(self.i_j, 1)  # none*len*1

        self.l2_loss += tf.nn.l2_loss(self.Wau)
        self.l2_loss += tf.nn.l2_loss(self.Wru)
        self.l2_loss += tf.nn.l2_loss(self.Wri)
        self.l2_loss += tf.nn.l2_loss(self.Wai)

        u_feas = tf.reduce_sum(tf.multiply(self.u_a, h_pool_flat_u), 1)
        i_feas = tf.reduce_sum(tf.multiply(self.i_a, h_pool_flat_i), 1)

        self.U_to_latent_W = tf.Variable(tf.random_uniform([num_filters, n_latent], -0.1, 0.1), name='U_to_latent')
        self.U_to_latent_b = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="U_to_latent_b")
        self.I_to_latent_W = tf.Variable(tf.random_uniform([num_filters, n_latent], -0.1, 0.1), name='I_to_latent_W')
        self.I_to_latent_b = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="I_to_latent_b")
        u_feas = tf.nn.tanh(tf.matmul(u_feas, self.U_to_latent_W)+self.U_to_latent_b)
        i_feas = tf.nn.tanh(tf.matmul(i_feas, self.I_to_latent_W)+self.I_to_latent_b)
        return u_feas, i_feas

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

    def rev_to_u_i(self, rnn_size, n_latent):
        w_to_user = tf.Variable(tf.random_normal([rnn_size, n_latent], stddev=0.1),name='w_to_user')
        w_to_item = tf.Variable(tf.random_normal([rnn_size, n_latent], stddev=0.1),name='w_to_item')
        b_to_user = tf.Variable(tf.constant(0.1, shape=[n_latent]), dtype=tf.float32, name="b_to_user")
        b_to_item = tf.Variable(tf.constant(0.1, shape=[n_latent]), dtype=tf.float32, name="b_to_item")
        u_fea_rev = tf.nn.tanh(tf.matmul(self.rev, w_to_user) + b_to_user)
        i_fea_rev = tf.nn.tanh(tf.matmul(self.rev, w_to_item) + b_to_item)
        return u_fea_rev, i_fea_rev

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


        # with tf.name_scope("attention"):
        #     W_atten_1 = tf.Variable(tf.random_uniform([rnn_size, rnn_size], -0.1, 0.1), name='W_atten_1')
        #     W_atten_2 = tf.Variable(tf.random_uniform([rnn_size, 1], -0.1, 0.1), name='W_atten_2')
        #     b_a_1 = tf.Variable(tf.constant(0.1, shape=[rnn_size]), name="b_a_1")
        #     b_a_2 = tf.Variable(tf.constant(0.1, shape=[1]), name="b_a_2")
        #     doc_vecs = tf.unstack(drop_bi_rnn_out, num=batch_size)
        #     for i in range(batch_size):
        #         alpha = tf.nn.softmax(tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(doc_vecs[i], W_atten_1)+b_a_1),W_atten_2)+b_a_2))
        #         doc_vecs[i] = tf.matmul(tf.transpose(alpha), doc_vecs[i])
        # return tf.reshape(tf.stack(doc_vecs), [-1, rnn_size])

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

        # with tf.name_scope("attention"):
        #     W_atten_1 = tf.Variable(tf.random_uniform([rnn_size, rnn_size], -0.1, 0.1), name='W_atten_1')
        #     W_atten_2 = tf.Variable(tf.random_uniform([rnn_size, 1], -0.1, 0.1), name='W_atten_2')
        #     b_a_1 = tf.Variable(tf.constant(0.1, shape=[rnn_size]), name="b_a_1")
        #     b_a_2 = tf.Variable(tf.constant(0.1, shape=[1]), name="b_a_2")
        #     doc_vecs = tf.unstack(drop_bi_rnn_out, num=batch_size)
        #     for i in range(batch_size):
        #         alpha = tf.nn.softmax(tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(doc_vecs[i], W_atten_1)+b_a_1),W_atten_2)+b_a_2))
        #         doc_vecs[i] = tf.matmul(tf.transpose(alpha), doc_vecs[i])
        # return tf.reshape(tf.stack(doc_vecs), [-1, rnn_size])
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


