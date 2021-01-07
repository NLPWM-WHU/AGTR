import numpy as np
import pickle

def get_com_num(rev1, rev2):
    com_num = 0
    if len(rev1) == 0 or len(rev2) == 0:
        return 0
    for i in range(len(rev1)):
        for j in range(len(rev2)):
                if rev1[i] == rev2[j]:
                    com_num += 1
                    break
    return com_num

def calc_rouge_1(gen_revs, revs, rev_num):
    gen_revs = remove_pad_word(gen_revs)
    P_sum = 0
    R_sum = 0
    F1_sum = 0
    for i in range(rev_num):
        com_num = get_com_num(gen_revs[i], revs[i])
        R = com_num / len(revs[i]) if len(revs[i]) else 0
        R_sum += R
        P = com_num / len(gen_revs[i]) if len(gen_revs[i]) else 0
        P_sum += P
        if R == 0 and P == 0:
            F1 = 0
        else:
            F1 = 2 * (R * P) / (R + P)
        F1_sum += F1
    print("ROUGE_1: R, P, F1", R_sum/rev_num, P_sum/rev_num, F1_sum/rev_num)

def split_2_gram(revs):
    revs_2 = []
    for rev in revs:
        rev_2 = []
        if len(rev) > 1:
            for i in range(len(rev) - 1):
                rev_2.append([rev[i], rev[i + 1]])
        revs_2.append(rev_2)
    return revs_2

def calc_rouge_2(gen_revs, revs, rev_num):
    gen_revs = remove_pad_word(gen_revs)
    revs_2 = split_2_gram(revs)
    gen_revs_2 = split_2_gram(gen_revs)
    P_sum = 0
    R_sum = 0
    F1_sum = 0
    for i in range(rev_num):
        if len(revs_2[i]) > 1:
            com_num = get_com_num(gen_revs_2[i], revs_2[i])
        else:
            com_num = 0
        R = com_num / len(revs_2[i]) if len(revs_2[i]) else 0
        R_sum += R
        P = com_num / len(gen_revs_2[i]) if len(gen_revs_2[i]) else 0
        P_sum += P
        if R == 0 and P == 0:
            F1 = 0
        else:
            F1 = 2 * (R * P) / (R + P)
        F1_sum += F1
    print("ROUGE_2: R, P, F1", R_sum / rev_num, P_sum / rev_num, F1_sum / rev_num)

def LCS(s1, s2):
    size1 = len(s1) + 1
    size2 = len(s2) + 1
    # 程序多加一行，一列，方便后面代码编写
    chess = [[["", 0] for j in list(range(size2))] for i in list(range(size1))]
    for i in list(range(1, size1)):
        chess[i][0][0] = s1[i - 1]
    for j in list(range(1, size2)):
        chess[0][j][0] = s2[j - 1]

    for i in list(range(1, size1)):
        for j in list(range(1, size2)):
            if s1[i - 1] == s2[j - 1]:
                chess[i][j] = ['↖', chess[i - 1][j - 1][1] + 1]
            elif chess[i][j - 1][1] > chess[i - 1][j][1]:
                chess[i][j] = ['←', chess[i][j - 1][1]]
            else:
                chess[i][j] = ['↑', chess[i - 1][j][1]]
    i = size1 - 1
    j = size2 - 1
    s3 = []
    while i > 0 and j > 0:
        if chess[i][j][0] == '↖':
            s3.append(chess[i][0][0])
            i -= 1
            j -= 1
        if chess[i][j][0] == '←':
            j -= 1
        if chess[i][j][0] == '↑':
            i -= 1

    return len(s3)

def calc_rouge_L(gen_revs, revs, rev_num):
    gen_revs = remove_pad_word(gen_revs)
    P_sum = 0
    R_sum = 0
    F1_sum = 0
    for i in range(rev_num):
        lcs_len = LCS(revs[i], gen_revs[i])
        R = lcs_len / len(revs[i]) if len(revs[i]) else 0
        R_sum += R
        P = lcs_len / len(gen_revs[i]) if len(gen_revs[i]) else 0
        P_sum += P
        if R == 0 and P == 0:
            F1 = 0
        else:
            F1 = 2 * (R * P) / (R + P)
        F1_sum += F1
    print("ROUGE_L: R, P, F1", R_sum / rev_num, P_sum / rev_num, F1_sum / rev_num)

def slpit_SU4(revs):
    new_revs = []
    for rev in revs:
        new_rev = []
        rev_len = len(rev)
        for i in range(rev_len):
            for j in range(1, 6):
                if i + j < rev_len:
                    new_rev.append([rev[i], rev[i + j]])
        new_revs.append(new_rev)
    return new_revs

def calc_rouge_SU4(gen_revs, revs, rev_num):
    gen_revs = remove_pad_word(gen_revs)
    P_sum = 0
    R_sum = 0
    F1_sum = 0
    g_r_2 = slpit_SU4(gen_revs)
    r_2 = slpit_SU4(revs)
    for i in range(rev_num):
        com_num_2 = get_com_num(g_r_2[i], r_2[i])
        com_num_1 = get_com_num(gen_revs[i], revs[i])
        R = (com_num_2 + com_num_1) / (len(revs[i]) + len(r_2[i])) if (len(revs[i]) + len(r_2[i])) else 0
        R_sum += R
        P = (com_num_2 + com_num_1) / (len(g_r_2[i]) + len(gen_revs[i])) if (len(g_r_2[i]) + len(gen_revs[i])) else 0
        P_sum += P
        if R == 0 and P == 0:
            F1 = 0
        else:
            F1 = 2 * (R * P) / (R + P)
        F1_sum += F1
    print("ROUGE_SU4: R, P, F1", R_sum / rev_num, P_sum / rev_num, F1_sum / rev_num)

def remove_pad_word(revs):
    new_revs = []
    for rev in revs:
        new_rev = []
        for id in rev:
            if id != 0:
                new_rev.append(id)
        new_revs.append(new_rev)
    del revs
    return new_revs

dataset = "Grocery_and_Gourmet_Food"
TPS_DIR = '../common_data/' + dataset + '/'
gen_revs_file = TPS_DIR + "NRT_gen_revs.txt"
val_data_file = TPS_DIR + "/val_trans_sl40_mf10"

batch_size = 64

with open(gen_revs_file, 'rb') as pkl_file:
    gen_revs = pickle.load(pkl_file)

with open(val_data_file, 'rb') as pkl_file:
    val_data = pickle.load(pkl_file)


val_batchs = int(len(val_data)/batch_size)
val_data = val_data[:val_batchs*batch_size]

uids, iids, ys, revs = zip(*val_data)
del uids, iids, ys

rev_num = len(revs)
rev_len = len(revs[0])

revs = remove_pad_word(revs)

calc_rouge_1(gen_revs, revs, rev_num)
calc_rouge_2(gen_revs, revs, rev_num)
calc_rouge_L(gen_revs, revs, rev_num)
#calc_rouge_SU4(gen_revs, revs, rev_num)


