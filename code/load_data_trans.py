'''
Data pre process

@author:
Chong Chen (cstchenc@163.com)

@ created:
25/8/2017
@references:
'''
import os
import json
import pandas as pd
import pickle
import numpy as np
from tensorflow.contrib import learn
import random

#dataset = 'Digital_Music'
#dataset = 'Toys_and_Games'
#dataset = 'Musical_Instruments'
dataset = 'Automotive'
#dataset = "Patio_Lawn_and_Garden"
#dataset = "Grocery_and_Gourmet_Food"
#dataset = "yelp2017"

TPS_DIR = '../common_data/' + dataset + '/'
TP_file = TPS_DIR + dataset + '_5.json'
if dataset == "yelp2017":
    TP_file = TPS_DIR + dataset + '_KCORE20.json'
stop_words_file = "../common_data/stop_words.txt"

train_data_file = TPS_DIR + "train_trans_sl40_mf10_nodrop"
val_data_file = TPS_DIR + "val_trans_sl40_mf10_nodrop"
para_data_file = TPS_DIR + "para_trans_sl40_mf10_nodrop"

min_fre = 10
rev_len_max = 40

def drop_Punctuation(line):
    out = ""
    for c in line:
        if c.isalpha():
            out += c
        else:
            out += " "
    return out

def drop_words(words, stop_words_file=stop_words_file):
    stop_words = []
    with open(stop_words_file, 'r') as f:
        for line in f:
            stop_words.extend(line.split())
    out = []
    for word in words:
        if len(word) > 2 and word not in stop_words:
            out.append(word)
    return out

def encode_text(rev_list):
    lens = []
    for rev in rev_list:
        lens.append(len(rev.split()))
    lens.sort()
    vocab = learn.preprocessing.VocabularyProcessor(rev_len_max, min_fre)
    text = np.array(list(vocab.fit_transform(rev_list)))
    num_vocab = len(vocab.vocabulary_)
    print("num vocab: ", num_vocab)
    return text, num_vocab, vocab.vocabulary_._mapping, rev_len_max

def numerize(users_id, items_id):
    uid = []
    iid = []
    for x in users_id:
        uid.append(user2id[x])
    for x in items_id:
        iid.append(item2id[x])
    del users_id, items_id
    return uid, iid

def read_amazon_data():
    with open(TP_file) as f:
        for line in f:
            js = json.loads(line)
            if str(js['reviewerID']) == 'unknown':
                print("unknown")
                continue
            if str(js['asin']) == 'unknown':
                print("unknown2")
                continue
            #review = ' '.join(drop_words(drop_Punctuation(js['reviewText'].lower()).split()))
            review = ' '.join((drop_Punctuation(js['reviewText'].lower()).split()))
            reviews.append(review)
            users_id.append(str(js['reviewerID']) + ',')
            items_id.append(str(js['asin']) + ',')
            ratings.append(str(js['overall']))

def read_yelp_data():
    with open(TP_file) as f:
        for line in f:
            js = json.loads(line)
            if str(js['user_id']) == 'unknown':
                print("unknown")
                continue
            if str(js['business_id']) == 'unknown':
                print("unknown2")
                continue
            review = ' '.join(drop_words(drop_Punctuation(js['text'].lower()).split()))
            #review = ' '.join((drop_Punctuation(js['text'].lower()).split()))
            reviews.append(review)
            users_id.append(str(js['user_id']) + ',')
            items_id.append(str(js['business_id']) + ',')
            ratings.append(str(js['stars']))

def get_data(users_id, items_id, ratings, reviews, ids):
    uid = []
    iid = []
    rate = []
    rev = []
    uid_left = []
    iid_left = []
    rate_left = []
    rev_left = []
    for id, num in enumerate(ids):
        if num:
            uid.append(users_id[id])
            iid.append(items_id[id])
            rate.append(ratings[id])
            rev.append(reviews[id])
        else:
            uid_left.append(users_id[id])
            iid_left.append(items_id[id])
            rate_left.append(ratings[id])
            rev_left.append(reviews[id])
    del users_id, items_id, ratings, reviews
    return uid, iid,  rate, rev, uid_left, iid_left, rate_left, rev_left

np.random.seed(2017)

users_id = []
items_id = []
ratings = []
reviews = []

if dataset == "yelp2017":
    read_yelp_data()
else:
    read_amazon_data()

unique_uid = sorted(set(users_id))
unique_iid = sorted(set(items_id))
item2id = dict((iid, i) for (i, iid) in enumerate(unique_iid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
id2item = dict((i, iid) for (i, iid) in enumerate(unique_iid))
id2user = dict((i, uid) for (i, uid) in enumerate(unique_uid))
user_num = len(user2id)
item_num = len(item2id)
data_num = len(users_id)
print('user_num', user_num)
print('item_num', item_num)
print('data_num', data_num)

users_id, items_id = numerize(users_id, items_id)

reviews, num_vocab, vocabs, rev_len = encode_text(reviews)

n_ratings = data_num
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

uid_test_val, iid_test_val,  rate_test_val, rev_test_val, uid_left, iid_left, rate_left, rev_left \
    = get_data(users_id, items_id, ratings, reviews, test_idx)
data_train = list(zip(uid_left, iid_left, rate_left, rev_left))

n_ratings = len(uid_test_val)
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

uid_test, iid_test,  rate_test, rev_test, uid_left, iid_left, rate_left, rev_left = get_data(uid_test_val,
                                            iid_test_val, rate_test_val, rev_test_val, test_idx)
data_valid = list(zip(uid_left, iid_left, rate_left, rev_left))


para = {}
para["num_vocab"] = num_vocab
para["vocab"] = vocabs
para["rev_len"] = rev_len
para["user_num"] = user_num
para["item_num"] = item_num

pickle.dump(data_train, open(train_data_file, "wb"))
pickle.dump(data_valid, open(val_data_file, "wb"))
pickle.dump(para, open(para_data_file, "wb"))
















