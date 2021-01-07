import numpy as np

class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data):
        self.num_batch = int(len(data) / self.batch_size)
        uid, iid, y, rev = zip(*data)
        self.rev = rev[:self.num_batch * self.batch_size]
        self.uid = uid[:self.num_batch * self.batch_size]
        self.iid = iid[:self.num_batch * self.batch_size]
        self.y = y[:self.num_batch * self.batch_size]
        self.g_seq_batchs = np.split(np.array(self.rev), self.num_batch, 0)
        self.g_uid_batchs = np.split(np.array(self.uid), self.num_batch, 0)
        self.g_iid_batchs = np.split(np.array(self.iid), self.num_batch, 0)
        self.g_y_batchs = np.split(np.array(self.y), self.num_batch, 0)
        self.pointer = 0
        del uid, iid, y

    def next_batch(self):
        g_seq_batch = self.g_seq_batchs[self.pointer]
        g_uid_batch = self.g_uid_batchs[self.pointer]
        g_iid_batch = self.g_iid_batchs[self.pointer]
        g_y_batch = self.g_y_batchs[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return g_seq_batch, g_uid_batch, g_iid_batch, g_y_batch

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, train_data, negative_examples):
        # Load data
        uids, iids, rates, positive_examples = zip(*train_data)
        positive_examples = list(positive_examples)
        uids = list(uids)
        iids = list(iids)
        rates = list(rates)
        self.sentences = np.array(positive_examples + negative_examples)
        self.uids = np.array(uids + uids)
        self.iids = np.array(iids + iids)
        self.rates = np.array(rates + rates)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        self.uids = self.uids[shuffle_indices]
        self.iids = self.iids[shuffle_indices]
        self.rates = self.rates[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.uids = self.uids[:self.num_batch * self.batch_size]
        self.iids = self.iids[:self.num_batch * self.batch_size]
        self.rates = self.rates[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.uids_batches = np.split(self.uids, self.num_batch, 0)
        self.iids_batches = np.split(self.iids, self.num_batch, 0)
        self.rates_batches = np.split(self.rates, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer], self.uids_batches[self.pointer],\
              self.iids_batches[self.pointer], self.rates_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

