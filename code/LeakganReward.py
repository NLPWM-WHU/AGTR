import numpy as np


def redistribution(idx, total, min_v):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))


def rescale(reward, rollout_num=1.0):
    reward = np.array(reward)
    x, y = reward.shape
    ret = np.zeros((x, y))
    for i in range(x):
        l = reward[i]
        rescalar = {}
        for s in l:
            rescalar[s] = s
        idxx = 1
        min_s = 1.0
        max_s = 0.0
        for s in rescalar:
            rescalar[s] = redistribution(idxx, len(l), min_s)
            idxx += 1
        for j in range(y):
            ret[i, j] = rescalar[reward[i, j]]
    return ret

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

class Reward(object):
    def __init__(self, model_gen, model_pre, dis, sess, rollout_num):
        self.model = model_gen
        self.model_pre = model_pre
        self.sess = sess
        self.dis = dis
        self.rollout_num = rollout_num

    def get_reward(self, input_x,  uid_batch, iid_batch, u_batch, i_batch, reuid, reiid, r_oneh, rev_lens_batch):
        rewards = []
        for i in range(self.rollout_num):
            for given_num in range(1, self.model.sequence_length // self.model.step_size):
                real_given_num = given_num * self.model.step_size
                feed = {self.model.x: input_x, self.model.given_num: real_given_num, self.model.input_u:u_batch,self.model.input_i:i_batch,
                        self.model.input_reuid: reuid,self.model.input_reiid: reiid, self.model.input_uid: uid_batch,self.model.input_iid: iid_batch,
                        self.model.drop_out: 1.0, self.model.dropout_his:1.0, self.model.train:0, self.model.rate_emb_onehot:r_oneh, self.model.batch_lens:rev_lens_batch}
                samples = self.sess.run(self.model.gen_for_reward, feed)
                # print samples.shape
                feed = {self.dis.D_input_x: samples, self.dis.input_uid: uid_batch, self.dis.input_iid: iid_batch, self.dis.rate_emb_onehot:r_oneh}
                ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {self.dis.D_input_x: input_x, self.dis.input_uid: uid_batch, self.dis.input_iid: iid_batch, self.dis.rate_emb_onehot:r_oneh}
            ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.model.sequence_length // self.model.step_size - 1] += ypred

        #rewards = np.array(rewards)
        rewards = rescale(np.array(rewards), self.rollout_num)
        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)  # batch_size x seq_length
        return rewards

    def get_reward_rnn(self, input_x,  uid_batch, iid_batch):
        rewards = []
        for i in range(self.rollout_num):
            for given_num in range(1, self.model.sequence_length // self.model.step_size):
                real_given_num = given_num * self.model.step_size
                feed = {self.model.x: input_x, self.model.given_num: real_given_num, self.model.input_uid: uid_batch,self.model.input_iid: iid_batch,
                        self.model.drop_out: 1.0, self.model.train:0}
                samples = self.sess.run(self.model.gen_for_reward, feed)
                # print samples.shape
                feed = {self.dis.D_input_x: samples, self.dis.input_uid: uid_batch, self.dis.input_iid: iid_batch}
                ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {self.dis.D_input_x: input_x, self.dis.input_uid: uid_batch, self.dis.input_iid: iid_batch}
            ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.model.sequence_length // self.model.step_size - 1] += ypred

        #rewards = np.array(rewards)
        rewards = rescale(np.array(rewards), self.rollout_num)
        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)  # batch_size x seq_length
        return rewards

    def get_reward_rate(self, input_x,  uid_batch, iid_batch, u_batch, i_batch, reuid, reiid, y_batch):
        rewards = []
        for i in range(self.rollout_num):  #self.rollout_num
            for given_num in range(1, self.model.sequence_length // self.model.step_size):   #
                real_given_num = given_num * self.model.step_size
                feed = {self.model.x: input_x, self.model.given_num: real_given_num, self.model.input_u: u_batch,
                                self.model.input_i: i_batch,
                                self.model.input_reuid: reuid, self.model.input_reiid: reiid, self.model.input_uid: uid_batch,
                                self.model.input_iid: iid_batch,
                                self.model.drop_out: 1.0, self.model.dropout_his: 1.0, self.model.train: 0}
                samples = self.sess.run(self.model.gen_x, feed_dict=feed)
                feed = {self.model_pre.input_u: u_batch,
                                self.model_pre.input_i: i_batch,
                                self.model_pre.input_uid: uid_batch,
                                self.model_pre.input_iid: iid_batch,
                                self.model_pre.input_y: y_batch,
                                self.model_pre.input_reuid: reuid,
                                self.model_pre.input_reiid: reiid,
                                self.model_pre.input_rev: samples,
                                self.model_pre.dropout: 1.0}
                loss = self.sess.run(self.model_pre.reward_loss, feed)
                loss = 1 - np.array(loss) / (1.0 * 25)
                if i == 0:
                    rewards.append(loss)
                else:
                    rewards[given_num - 1] += loss
            feed = {self.model_pre.input_u: u_batch,
                self.model_pre.input_i: i_batch,
                self.model_pre.input_uid: uid_batch,
                self.model_pre.input_iid: iid_batch,
                self.model_pre.input_y: y_batch,
                self.model_pre.input_reuid: reuid,
                self.model_pre.input_reiid: reiid,
                self.model_pre.input_rev: input_x,
                self.model_pre.dropout: 1.0}
            loss = self.sess.run(self.model_pre.reward_loss, feed)
            loss = 1 - np.array(loss) / (1.0 * 25)
            if i == 0:
                rewards.append(loss)
            else:
                rewards[self.model.sequence_length // self.model.step_size - 1] += loss
        #rewards = np.array(rewards)
        rewards = rescale(np.array(rewards),  self.rollout_num)
        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)  # batch_size x seq_length
        return rewards

    def get_reward_rate_woker(self, input_x,  uid_batch, iid_batch, u_batch, i_batch, reuid, reiid, y_batch):
        rewards = []
        for i in range(self.rollout_num):  #self.rollout_num
            for given_num in range(1, self.model.sequence_length):   #
                real_given_num = given_num * self.model.step_size
                feed = {self.model.x: input_x, self.model.given_num: real_given_num, self.model.input_u: u_batch,
                                self.model.input_i: i_batch,
                                self.model.input_reuid: reuid, self.model.input_reiid: reiid, self.model.input_uid: uid_batch,
                                self.model.input_iid: iid_batch,
                                self.model.drop_out: 1.0, self.model.dropout_his: 1.0, self.model.train: 0}
                samples = self.sess.run(self.model.gen_x, feed_dict=feed)
                feed = {self.model_pre.input_u: u_batch,
                                self.model_pre.input_i: i_batch,
                                self.model_pre.input_uid: uid_batch,
                                self.model_pre.input_iid: iid_batch,
                                self.model_pre.input_y: y_batch,
                                self.model_pre.input_reuid: reuid,
                                self.model_pre.input_reiid: reiid,
                                self.model_pre.input_rev: samples,
                                self.model_pre.dropout: 1.0}
                loss = self.sess.run(self.model_pre.reward_loss, feed)
                loss = 1 - np.array(loss) / (1.0 * 25)
                if i == 0:
                    rewards.append(loss)
                else:
                    rewards[given_num - 1] += loss
            feed = {self.model_pre.input_u: u_batch,
                self.model_pre.input_i: i_batch,
                self.model_pre.input_uid: uid_batch,
                self.model_pre.input_iid: iid_batch,
                self.model_pre.input_y: y_batch,
                self.model_pre.input_reuid: reuid,
                self.model_pre.input_reiid: reiid,
                self.model_pre.input_rev: input_x,
                self.model_pre.dropout: 1.0}
            loss = self.sess.run(self.model_pre.reward_loss, feed)
            loss = 1 - np.array(loss) / (1.0 * 25)
            if i == 0:
                rewards.append(loss)
            else:
                rewards[self.model.sequence_length - 1] += loss
        rewards = np.array(rewards)
        #rewards = rescale(np.array(rewards),  self.rollout_num)
        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)  # batch_size x seq_length
        return rewards
