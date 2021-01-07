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
import random

#dataset = 'Digital_Music'
#dataset = 'Toys_and_Games'
#dataset = 'Musical_Instruments'
#dataset = 'Patio_Lawn_and_Garden'
dataset = 'Automotive'
#dataset = "Grocery_and_Gourmet_Food"
#dataset = "yelp2017"


TPS_DIR = '../common_data/' + dataset + '/'
TP_file = TPS_DIR + dataset + '_5.json'
if dataset == "yelp2017":
    TP_file = TPS_DIR + dataset + '_KCORE20.json'
stop_words_file = "../common_data/stop_words.txt"

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

def price_pro(price):
    if price == 0:
        return 0
    elif price < 10:
        return 1
    elif price < 40:
        return 2
    elif price < 100:
        return 3
    elif price < 200:
        return 4
    elif price < 350:
        return 5
    else:
        return 6

def rank_pro(rank):
    if rank < 1000:
        return 0
    elif rank<3000:
        return 1
    elif rank<6000:
        return 2
    elif rank<10000:
        return 3
    elif rank<15000:
        return 4
    elif rank<21000:
        return 5
    elif rank<28000:
        return 6
    elif rank<36000:
        return 7
    elif rank<45000:
        return 8
    elif rank<55000:
        return 9
    else:
        return 10

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
            review = ' '.join(drop_words(drop_Punctuation(js['reviewText'].lower()).split()))
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
            reviews.append(review)
            users_id.append(str(js['user_id']) + ',')
            items_id.append(str(js['business_id']) + ',')
            ratings.append(str(js['stars']))

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

all_data_num = len(users_id)
print('all data num', len(users_id))

users_id, items_id = numerize(users_id, items_id)

data = pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews)})[['user_id', 'item_id', 'ratings', 'reviews']]

tp_rating = data[['user_id','item_id','ratings']]
n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train= tp_rating[~test_idx]

data2 = data[test_idx]
data = data[~test_idx]


n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]
tp_train.to_csv(os.path.join(TPS_DIR, 'train.csv'), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'valid.csv'), index=False,header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'test.csv'), index=False,header=None)

user_reviews = {}
item_reviews = {}
user_rates = {}
item_rates = {}
user_rid = {}
item_rid = {}
for i in data.values:
    if i[0] in user_reviews:
        user_reviews[i[0]].append(i[3])
        user_rates[i[0]].append(float(i[2]))
        user_rid[i[0]].append(i[1])
    else:
        user_rid[i[0]] = [i[1]]
        user_reviews[i[0]] = [i[3]]
        user_rates[i[0]] = [float(i[2])]
    if i[1] in item_reviews:
        item_reviews[i[1]].append(i[3])
        item_rates[i[1]].append(float(i[2]))
        item_rid[i[1]].append(i[0])
    else:
        item_reviews[i[1]] = [i[3]]
        item_rates[i[1]] = [float(i[2])]
        item_rid[i[1]] = [i[0]]

for i in data2.values:
    if i[0] in user_reviews:
        l = 1
    else:
        user_rid[i[0]] = [0]
        user_reviews[i[0]] = ['0']
        user_rates[i[0]] = [3.0]
    if i[1] in item_reviews:
        l = 1
    else:
        item_reviews[i[1]] = [0]
        item_rid[i[1]] = ['0']
        item_rates[i[1]] = [3.0]

pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))
pickle.dump(user_rates, open(os.path.join(TPS_DIR, 'user_rate'), 'wb'))
pickle.dump(item_rates, open(os.path.join(TPS_DIR, 'item_rate'), 'wb'))











