from SeqganDiscriminator import Discriminator
from SeqganGenerator import Generator
from SeqganReward import Reward
from tarmf_smi import Rev_gen_pre
import numpy as np
import tensorflow as tf
import random
import pickle
from dataloader import Gen_Data_loader, Dis_dataloader


def read_para_data(para_data_file):
    print("Loading data...")
    pkl_file = open(para_data_file, 'rb')
    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    review_len_u = para['review_len_u']
    review_len_i = para['review_len_i']
    vocab_user = para['user_vocab']
    vocab_item = para['item_vocab']
    u_text = para['u_text']
    i_text = para['i_text']
    u_rids = para['u_rids']
    i_rids = para['i_rids']
    print('user_num', user_num)
    print('item_num', item_num)
    print('review_num_u', review_num_u)
    print('review_len_u', review_len_u)
    print('review_num_i', review_num_i)
    print('review_len_i', review_len_i)

    pkl_file = open(para_trans_file, 'rb')
    para = pickle.load(pkl_file)
    vocab = para['vocab']
    rev_len = para["rev_len"]
    rev_vocab_num = para["num_vocab"]

    print("rev_num_vocab", rev_vocab_num)
    print("rev_len", rev_len)
    return user_num, item_num, review_num_u, review_num_i, review_len_u, review_len_i, vocab_user, vocab_item, u_text, \
           i_text, u_rids, i_rids, vocab, rev_len, rev_vocab_num

def shuffle_train_data(train_data):
    indexs = np.arange(len(train_data))
    np.random.shuffle(indexs)
    new_data = []
    for index in indexs:
        new_data.append(train_data[index])
    del indexs
    return new_data

def get_feed_dict(deep, u_batch, i_batch, uid, iid, reuid, reiid, y_batch, rev_batch, drop=0.5):
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.input_rev: rev_batch,
        deep.dropout: drop,
    }
    return feed_dict

def drop_u_i_his_rev_special(uid, iid, u_batch, i_batch, reuid, reiid):
    # drop u
    for i in range(batch_size):
        for j in range(len(reuid[0])):
            if reuid[i][j] == iid[i]:
                u_batch[i][j] = np.array([1]*len(u_batch[i][j]))

    for i in range(batch_size):
        for j in range(len(reiid[0])):
            if reiid[i][j] == uid[i]:
                i_batch[i][j] = np.array([1]*len(i_batch[i][j]))
    return u_batch, i_batch

def get_batch(uid, iid, u_text, i_text, u_rids, i_rids):
    u_batch = []
    i_batch = []
    reuid = []
    reiid = []
    for i in range(len(uid)):
        u_batch.append(u_text[uid[i]])
        i_batch.append(i_text[iid[i]])
        reuid.append(u_rids[uid[i]])
        reiid.append(i_rids[iid[i]])
    u_batch = np.array(u_batch)
    i_batch = np.array(i_batch)
    reuid = np.array(reuid)
    reiid = np.array(reiid)
    #u_batch, i_batch = drop_u_i_his_rev_special(uid, iid, u_batch, i_batch, reuid, reiid)
    return u_batch, i_batch, reuid, reiid

def fill_u_i_matrix(ids, feas, matrix):
    for i in range(len(feas)):
        matrix[ids[i]] = feas[i]

def train(sess, deep, train_data, u_text, i_text, u_rids, i_rids, train_op, train_able=True):
    train_mae = 0
    train_mse = 0
    train_loss = 0
    shuffled_data = shuffle_train_data(train_data)
    ll = int(len(shuffled_data) / batch_size)
    data_size_train = len(shuffled_data)
    for batch_num in range(ll):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size_train)
        data_train = shuffled_data[start_index:end_index]
        uid, iid, y_batch, rev_batch = zip(*data_train)
        u_batch, i_batch, reuid, reiid = get_batch(uid, iid, u_text, i_text, u_rids, i_rids)
        feed_dict = get_feed_dict(deep, u_batch, i_batch, uid, iid, reuid, reiid, y_batch, rev_batch,0.5)
        if train_able:   # train
            _, loss, t_mse, t_mae = sess.run([train_op, deep.loss, deep.mse, deep.mae], feed_dict)
            train_mse += t_mse
            train_mae += t_mae
            train_loss += loss
            train_rmse = np.sqrt(train_mse / ll)
        else:
            u_feas, i_feas = sess.run([deep.u_feas_his, deep.i_feas_his], feed_dict)   #get_u_i_matrix
            fill_u_i_matrix(uid, u_feas, user_matrix)
            fill_u_i_matrix(iid, i_feas, item_matrix)
    if train_able:
        print("tra:loss_3:", train_loss / ll, "mae:", train_mae / ll, "rmse:", train_rmse, "mse:", train_mse / ll)


def val(sess, deep, val_data, data_size_val, u_text, i_text, u_rids, i_rids):
    loss_s = 0
    mae_s = 0
    mse_s = 0
    ll_test = int(len(val_data) / batch_size)
    for batch_num in range(ll_test):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size_val)
        data_val = val_data[start_index:end_index]
        uid_valid, iid_valid, y_batch, rev_batch = zip(*data_val)
        u_valid, i_valid, reuid, reiid = get_batch(uid_valid, iid_valid, u_text, i_text, u_rids, i_rids)
        feed_dict = get_feed_dict(deep, u_valid, i_valid,  uid_valid, iid_valid, reuid, reiid, y_batch, rev_batch, 1.0)
        loss, mae, mse = sess.run([deep.loss, deep.mae, deep.mse], feed_dict)
        loss_s = loss_s + loss
        mae_s = mae_s + mae
        mse_s = mse_s + mse
    rmse_s = np.sqrt(mse_s / ll_test)

    print("val:loss:",loss_s / ll_test, "mae:", mae_s / ll_test,  "rmse:", rmse_s , "mse:", mse_s / ll_test)
    return loss_s / ll_test,  mae_s / ll_test, rmse_s, mse_s / ll_test

def dis_assign_U_I_matrix(model, u_m, i_m):
    model.uidW.assign(u_m)
    model.iidW.assign(i_m)

def gen_assign_U_I_matrix(model, u_m, i_m):
    model.uidW.assign(u_m)
    model.iidW.assign(i_m)

def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()
    for it in range(data_loader.num_batch):
        rev_batch, uid_batch, iid_batch, y_batch = data_loader.next_batch()
        g_loss = trainable_model.pretrain_step(sess, rev_batch, uid_batch, iid_batch)
        supervised_g_losses.append(g_loss)
    return np.mean(supervised_g_losses)

def target_loss(sess, model, data_loader):
    nll = []
    data_loader.reset_pointer()
    for it in range(data_loader.num_batch):
        x_batch, uid_batch, iid_batch, y_batch = data_loader.next_batch()
        g_loss = sess.run(model.pretrain_loss, {model.x: x_batch, model.input_uid: uid_batch,model.input_iid: iid_batch,})
        nll.append(g_loss)
    return np.mean(nll)

def get_negative_data(sess, data_loader, model):
    g_samples = []
    data_loader.reset_pointer()
    for it in range(data_loader.num_batch):
        x_batch, uid_batch, iid_batch, y_batch = data_loader.next_batch()
        feed_dict= {model.x: x_batch, model.input_uid: uid_batch,model.input_iid: iid_batch}
        gen_x = sess.run(generator.gen_x, feed_dict=feed_dict)
        g_samples.extend(gen_x)
    return g_samples


def pre_train_gen(sess, generator, gen_data_loader, likelihood_data_loader, pre_epoch_num, u_text, i_text, u_rids,
                  i_rids, saver, best_mae, best_rmse, best_mse, best_nll):
    error_num_max = 5
    error_num = 0
    true_epoch = 0
    print("pre-gen train loss:")
    for epoch in range(pre_epoch_num):
        true_epoch += 1
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        test_loss = target_loss(sess, generator, likelihood_data_loader)
        if test_loss < best_nll:
            best_nll = test_loss
            saver.save(sess, checkpoint_file)
            gen_test_revs = get_negative_data(sess, likelihood_data_loader, generator)
            pickle.dump(gen_test_revs, open(gen_revs_file, "wb"))
        print("best nll:", best_nll)
        print("epoch", epoch, "train_loss:", loss, "test_loss:", test_loss)

    saver.restore(sess, checkpoint_file)
    return true_epoch-error_num_max, best_mae, best_rmse, best_mse, best_nll

def train_discriminator(sess, gen_data_loader, dis_data_loader, generator, discriminator, u_text, i_text, u_rids, i_rids, train_data, pre_epoch_num,
                        best_mae, best_rmse, best_mse, best_nll):
    for epoch in range(pre_epoch_num):
        negative_data = get_negative_data(sess, gen_data_loader, generator)
        dis_data_loader.load_train_data(train_data[:len(negative_data)], negative_data)
        for _ in range(3):
            dis_data_loader.next_batch()
            x_batch, y_batch, uid_batch, iid_batch, rate_batch = dis_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.input_uid: uid_batch,
                discriminator.input_iid: iid_batch
            }
            loss, _ = sess.run([discriminator.d_loss, discriminator.train_op], feed)

    test_loss = target_loss(sess, generator, likelihood_data_loader)
    if test_loss < best_nll:
        best_nll = test_loss
        gen_test_revs = get_negative_data(sess, likelihood_data_loader, generator)
        pickle.dump(gen_test_revs, open(gen_revs_file, "wb"))
    print("best nll:", best_nll)

    return best_mae, best_rmse, best_mse, best_nll

def generate_batch_data(sess, data_loader, generator):
    x_batch, uid_batch, iid_batch, y_batch = data_loader.next_batch()
    #gen_x = sess.run(generator.gen_x, feed_dict={generator.x: x_batch, generator.input_uid: uid_batch, generator.input_iid: iid_batch})
    gen_x = sess.run(generator.gen_x, feed_dict={generator.input_uid: uid_batch, generator.input_iid: iid_batch})
    return gen_x, uid_batch, iid_batch

def aderv_gan(adversarial_epoch_num, sess, generator, discriminator, reward, gen_data_loader, dis_data_loader, u_text,
              i_text, u_rids, i_rids, train_data, best_mae, best_rmse, best_mse, best_nll):
    gen_data_loader.reset_pointer()
    for epoch in range(adversarial_epoch_num):
        print('epoch:' + str(epoch))
        x_batch, uid_batch, iid_batch, y_batch = gen_data_loader.next_batch()
        feed_dict = {generator.x: x_batch, generator.input_uid: uid_batch, generator.input_iid: iid_batch}
        samples = sess.run(generator.gen_x, feed_dict=feed_dict)
        rewards = reward.get_reward(sess, samples, 16, discriminator, uid_batch, iid_batch)
        feed = {generator.x: samples, generator.rewards: rewards, generator.input_uid: uid_batch, generator.input_iid: iid_batch,}
        loss, _ = sess.run([generator.g_loss, generator.g_updates], feed_dict=feed)
        reward.update_params()

        test_loss = target_loss(sess, generator, likelihood_data_loader)
        if test_loss < best_nll:
            best_nll = test_loss
        print("best nll:", best_nll)

        best_mae, best_rmse, best_mse, best_nll = train_discriminator(sess, gen_data_loader, dis_data_loader, generator, discriminator, u_text, i_text, u_rids,
                            i_rids, train_data, 5, best_mae, best_rmse, best_mse, best_nll)   #15
        print("best_gan", "mse:", best_mae, "rmse:", best_rmse, "mse:", best_mse)

        if epoch % 10 == 0 and epoch != 0:
            train_epoch, best_mae, best_rmse, best_mse, best_nll = pre_train_gen(sess, generator, gen_data_loader, likelihood_data_loader, 5, u_text, i_text, u_rids,
                  i_rids, saver, best_mae, best_rmse, best_mse, best_nll)
            best_mae, best_rmse, best_mse, best_nll = train_discriminator(sess, gen_data_loader, dis_data_loader, generator, discriminator, u_text, i_text, u_rids,i_rids,
                                train_data, train_epoch, best_mae, best_rmse, best_mse, best_nll)
            print("best_gan", "mse:", best_mae, "rmse:", best_rmse, "mse:", best_mse)

if __name__ == '__main__':
    tf.flags.DEFINE_boolean('restore', False, 'Training or testing a model')
    tf.flags.DEFINE_boolean('resD', False, 'Training or testing a D model')
    tf.flags.DEFINE_string('model', "", 'Model NAME')
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    FLAGS = tf.flags.FLAGS

    dataset = "yelp2017"
    print('dataset', dataset, 'trans')

    TPS_DIR = '../common_data/' + dataset + '/'

    word_emb_size = 32
    word_vecs_file = "../common_data/glove.6B." + str(word_emb_size) + "d.txt"
    train_data_file = TPS_DIR + "/train_trans_sl40_mf10"
    val_data_file = TPS_DIR + "/val_trans_sl40_mf10"
    para_trans_file = TPS_DIR + "/para_trans_sl40_mf10"
    para_data_file = TPS_DIR + "/para_co"

    gen_revs_file = TPS_DIR + "gen_revs_tarmf_sl40_mf10.txt"
    checkpoint_file = TPS_DIR + "model_rev_gen_pre.ckpt"

    # u_i_revs_his parameter
    embedding_dim = word_emb_size
    filter_size_his = 3
    num_filters_his = 32   #
    rnn_size = 32    #
    dropout_keep_prob_his = 0.5
    l2_reg_lambda_his = 0.001
    n_latent = 32
    lr = 0.002
    co_iters = 1

    # generators
    emb_dim = 32
    hidden_dim = 32
    filter_size = [2, 3]
    num_filters = [100, 200]
    l2_reg_lambda = 0.2
    dropout_keep_prob = 0.75
    batch_size = 64
    dis_emb_dim = 64
    goal_size = 16
    goal_out_size = sum(num_filters)
    step_size = 4
    num_classes = 2

    SEED = 2017
    random.seed(SEED)
    np.random.seed(SEED)

    user_num, item_num, rev_num_u, rev_num_i, rev_len_u, rev_len_i, vocab_user, vocab_item, u_text, \
    i_text, u_rids, i_rids, vocab, rev_len, rev_vocab_num = read_para_data(para_data_file)

    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    tf.set_random_seed(2017)

    deep = Rev_gen_pre(rev_num_u=rev_num_u, rev_num_i=rev_num_i, rev_len_u=rev_len_u, rev_len_i=rev_len_i, rev_len=rev_len,
        user_num=user_num, item_num=item_num, user_vocab_size=len(vocab_user),item_vocab_size=len(vocab_item),
        n_latent=n_latent, word_emb_size=word_emb_size, filter_sizes=filter_size_his, num_filters_his=num_filters_his,
        batch_size=batch_size, rnn_size=rnn_size)

    train_op = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)

    seq_len = rev_len
    gen_data_loader = Gen_Data_loader(batch_size)
    likelihood_data_loader = Gen_Data_loader(batch_size)  # For testing
    dis_data_loader = Dis_dataloader(batch_size)

    discriminator = Discriminator(sequence_length=seq_len, num_classes=num_classes, vocab_size=rev_vocab_num,emd_dim=dis_emb_dim,
                filter_sizes=filter_size, num_filters=num_filters,user_num=user_num, item_num=item_num, n_latent=n_latent,
                                  l2_reg_lambda=l2_reg_lambda)
    generator = Generator(sequence_length=seq_len, num_vocabulary=rev_vocab_num,emb_dim=emb_dim, batch_size=batch_size,
                          hidden_dim=hidden_dim, item_num=item_num, user_num=user_num, n_latent=n_latent)

    sess.run(tf.global_variables_initializer())

    with open(train_data_file, 'rb') as pkl_file:
        train_data = pickle.load(pkl_file)

    with open(val_data_file, 'rb') as pkl_file:
        val_data = pickle.load(pkl_file)

    val_batchs = int(len(val_data)/batch_size)
    val_data = val_data[:val_batchs*batch_size]
    data_size_val = len(val_data)
    data_num = len(train_data)

    user_matrix = np.zeros([user_num, n_latent], float)
    item_matrix = np.zeros([item_num, n_latent], float)

    best_rmse_t = float("inf")
    best_mae_t = float("inf")
    best_mse_t = float("inf")

    best_rmse_v = float("inf")
    best_mae_v = float("inf")
    best_mse_v = float("inf")

    best_nll = float("inf")

    rate_pre_epochs = 30
    saver = tf.train.Saver()
    pre_epoch_num = 20   #80
    adversarial_epoch_num = 100
    for iter in range(co_iters):
        error_num_max = 5
        error_num = 0

        uids, iids, ys, revs = zip(*val_data)
        for epoch in range(rate_pre_epochs):
            print(str(epoch) + ':')
            train(sess, deep, train_data, u_text, i_text, u_rids, i_rids, train_op, train_able=True)
            loss_, mae_v, rmse_v, mse_v = val(sess, deep, val_data, data_size_val, u_text, i_text, u_rids, i_rids)
            if mae_v < best_mae_t:
                best_mae_t = mae_v
            if mse_v < best_mse_t:
                best_mse_t = mse_v

            if rmse_v < best_rmse_t:
                best_rmse_t = rmse_v
                saver.save(sess, checkpoint_file)
                print('best mae:', best_mae_t, "\n", 'best rmse:', best_rmse_t, "\n", 'best mse:', best_mse_t)
                error_num = 0
            else:
                error_num += 1
            if error_num == error_num_max:
                saver.restore(sess, checkpoint_file)
                break

        train(sess, deep, train_data, u_text, i_text, u_rids, i_rids, train_op, train_able=False)

        gen_assign_U_I_matrix(generator, user_matrix, item_matrix)
        dis_assign_U_I_matrix(discriminator, user_matrix, item_matrix)
        #
        # gan
        gen_data_loader.create_batches(train_data)
        likelihood_data_loader.create_batches(val_data)

        print('Start pre-training generator...')
        gen_pre_epoch, best_mae_v, best_rmse_v, best_mse_v, best_nll = pre_train_gen(sess, generator, gen_data_loader, likelihood_data_loader,
                    pre_epoch_num, u_text, i_text, u_rids,
                  i_rids, saver, best_mae_v, best_rmse_v, best_mse_v, best_nll)

        print('Start pre-training discriminator...')
        best_mae_v, best_rmse_v, best_mse_v, best_nll = train_discriminator(sess, gen_data_loader, dis_data_loader, generator, discriminator,
                                u_text, i_text, u_rids, i_rids, train_data, gen_pre_epoch, best_mae_v, best_rmse_v, best_mse_v, best_nll)

        reward = Reward(lstm=generator, update_rate=0.8, step_size=4)

        rr_epoch_num = 100

        print('Start Adversarial Training...')
        aderv_gan(adversarial_epoch_num, sess, generator, discriminator, reward, gen_data_loader, dis_data_loader,u_text,
                   i_text, u_rids, i_rids, train_data, best_mae_v, best_rmse_v, best_mse_v, best_nll)




