import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn
import pickle
import os
import re

#dataset = 'Digital_Music'
#dataset = 'Toys_and_Games'
#dataset = 'Musical_Instruments'
#dataset =  'Patio_Lawn_and_Garden'
dataset = 'Automotive'
#dataset = "Grocery_and_Gourmet_Food"
#dataset = "yelp2017"


TPS_DIR = '../common_data/' + dataset + '/'

valid_data = TPS_DIR + 'valid' + '.csv'
test_data = TPS_DIR + 'test' + '.csv'
train_data = TPS_DIR + 'train' + '.csv'
user_review = TPS_DIR + 'user_review'
item_review = TPS_DIR + 'item_review'
user_review_id = TPS_DIR + 'user_rid'
item_review_id = TPS_DIR + 'item_rid'
user_rate = TPS_DIR + 'user_rate'
item_rate = TPS_DIR + 'item_rate'
stopwords = "data/stopwords"

def pad_sentences(u_text, u_len, u2_len, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_num = u_len
    review_len = u2_len

    u_text2 = {}
    for i in u_text:
        u_reviews = u_text[i]
        padded_u_train = []
        for ri in range(review_num):
            if ri < len(u_reviews):
                sentence = u_reviews[ri]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append(new_sentence)
                else:
                    new_sentence = sentence[:review_len]
                    padded_u_train.append(new_sentence)
            else:
                new_sentence = [padding_word] * review_len
                padded_u_train.append(new_sentence)
        u_text2[i] = padded_u_train
    return u_text2


def pad_reviewid(u_rids, u_len, num):
    pad_u_train = []
    for i in range(len(u_rids)):
        x = u_rids[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_train.append(x)
    return pad_u_train


def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    print(len(vocabulary1), len(vocabulary2))
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    l = len(u_text)
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([[vocabulary_u[word] for word in words] for words in u_reviews])
        u_text2[i] = u
    l = len(i_text)
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([[vocabulary_i[word] for word in words] for words in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2


def encode_text(train_text, val_text):
    train_text.extend(val_text)
    lens = []
    for title in train_text:
        lens.append(len(title.split()))
    lens.sort()
    title_len = lens[int(0.9*len(lens))-1]
    print('title_len', title_len,)
    print('max_len', lens[-1], 'mean_len', sum(lens)/len(lens))
    vocab = learn.preprocessing.VocabularyProcessor(title_len)
    text = np.array(list(vocab.fit_transform(train_text)))
    num_vocab = len(vocab.vocabulary_)
    return text, num_vocab, vocab.vocabulary_._mapping, title_len

def pad_rate(rates, rate=3.0, rev_num=20):
    for index in rates:
        rate_list = rates[index]
        if len(rate_list) < rev_num:
            rates[index] = rate_list + [rate]*(rev_num-len(rate_list))
        else:
            rates[index] = rate_list[:rev_num]
    return rates

def load_data(train_data, valid_data, user_review, item_review, user_rid, item_rid, user_rate, item_rate):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train, iid_train,uid_valid, iid_valid,user_num, item_num \
        , user_rids, item_rids, user_rates, item_rates = \
        load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, user_rate, item_rate)
    print("load data done")
    u_text = pad_sentences(u_text, u_len, u2_len)
    user_rids = pad_reviewid(user_rids, u_len, item_num + 1)

    print ("pad user done")
    i_text = pad_sentences(i_text, i_len, i2_len)
    item_rids = pad_reviewid(item_rids, i_len, user_num + 1)

    user_rates = pad_rate(user_rates, 3.0, u_len)
    item_rates = pad_rate(item_rates, 3.0, i_len)
    print("pad rate done")

    print ("pad item done")
    user_voc = [xx for x in u_text.values() for xx in x]
    item_voc = [xx for x in i_text.values() for xx in x]

    vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
    print('vocabulary_user', len(vocabulary_user))
    print('vocabulary_item', len(vocabulary_item))
    u_text, i_text = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)

    return [u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item,
            vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, user_rids, item_rids, user_rates, item_rates]


def load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, user_rate, item_rate):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    f_train = open(train_data, "r")
    f1 = open(user_review, 'rb')
    f2 = open(item_review, 'rb')
    f3 = open(user_rid, 'rb')
    f4 = open(item_rid, 'rb')
    f5 = open(user_rate, 'rb')
    f6 = open(item_rate, 'rb')

    user_reviews = pickle.load(f1)
    item_reviews = pickle.load(f2)
    user_rids = pickle.load(f3)
    item_rids = pickle.load(f4)
    user_rates = pickle.load(f5)
    item_rates = pickle.load(f6)

    uid_train = []
    iid_train = []
    y_train = []
    u_text = {}
    i_text = {}
    i = 0
    for line in f_train:
        i = i + 1
        line = re.split('[,"\n]',line)
        # line = line.split(',')
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        if int(line[0]) not in u_text:
            u_text[int(line[0])] = []
            for s in user_reviews[int(line[0])]:
                s1 = s
                s1 = s1.split(" ")
                u_text[int(line[0])].append(s1)

        if int(line[1]) not in i_text:
            i_text[int(line[1])] = []
            for s in item_reviews[int(line[1])]:
                s1 = s
                s1 = s1.split(" ")
                i_text[int(line[1])].append(s1)
        y_train.append(float(line[2]))

    print("valid")
    uid_valid = []
    iid_valid = []
    y_valid = []
    f_valid = open(valid_data)
    for line in f_valid:
        line = re.split('[,"\n]', line)
        # line = line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        if int(line[0]) not in u_text:
            u_text[int(line[0])] = [['<PAD/>']]
        if int(line[1]) not in i_text:
            i_text[int(line[1])] = [['<PAD/>']]

        y_valid.append(float(line[2]))

    print("test")
    uid_test = []
    iid_test = []
    y_test = []
    f_test = open(test_data)
    for line in f_test:
        line = re.split('[,"\n]', line)
        uid_test.append(int(line[0]))
        iid_test.append(int(line[1]))
        if int(line[0]) not in u_text:
            u_text[int(line[0])] = [['<PAD/>']]

        if int(line[1]) not in i_text:
            i_text[int(line[1])] = [['<PAD/>']]

        y_test.append(float(line[2]))

    review_num_u = np.array([len(x) for x in u_text.values()])
    x = np.sort(review_num_u)
    u_len = x[int(0.9 * len(review_num_u)) - 1]
    review_len_u = np.array([len(j) for i in u_text.values() for j in i])
    x2 = np.sort(review_len_u)
    u2_len = x2[int(0.9 * len(review_len_u)) - 1]

    review_num_i = np.array([len(x) for x in i_text.values()])
    y = np.sort(review_num_i)
    i_len = y[int(0.9 * len(review_num_i)) - 1]
    review_len_i = np.array([len(j) for i in i_text.values() for j in i])
    y2 = np.sort(review_len_i)
    i2_len = y2[int(0.9 * len(review_len_i)) - 1]

    u_len = min(20, u_len)
    i_len = min(20, i_len)
    u2_len = min(50, u2_len)
    i2_len = min(50, i2_len)

    print("u_len:", u_len)
    print("i_len:", i_len)
    print("u2_len:", u2_len)
    print("i2_len:", i2_len)
    user_num = len(u_text)
    item_num = len(i_text)
    print("user_num:", user_num)
    print("item_num:", item_num)
    return [u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train, iid_train, uid_valid, iid_valid, user_num,
            item_num, user_rids, item_rids, user_rates, item_rates]

if __name__ == '__main__':
    u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item, uid_train, \
    iid_train, uid_valid, iid_valid, user_num, item_num,user_rids, item_rids, user_rates, item_rates\
        = load_data(train_data, valid_data, user_review, item_review, user_review_id, item_review_id, user_rate, item_rate)

    np.random.seed(2017)
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    batches_train = list(
        zip(userid_train, itemid_train,  y_train))
    batches_val = list(zip(userid_valid, itemid_valid,y_valid))
    print('write begin')
    output = open(os.path.join(TPS_DIR, 'train_co'), 'wb')
    pickle.dump(batches_train, output)
    output = open(os.path.join(TPS_DIR, 'val_co'), 'wb')
    pickle.dump(batches_val, output)

    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['review_num_u'] = u_text[0].shape[0]
    para['review_num_i'] = i_text[0].shape[0]
    para['review_len_u'] = u_text[1].shape[1]
    para['review_len_i'] = i_text[1].shape[1]
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['train_length'] = len(y_train)
    para['test_length'] = len(y_valid)
    para['u_text'] = u_text
    para['i_text'] = i_text
    para['u_rids'] = user_rids
    para['i_rids'] = item_rids
    para['u_rates'] = user_rates
    para['i_rates'] = item_rates

    output = open(os.path.join(TPS_DIR, 'para_co'), 'wb')
    pickle.dump(para, output)










