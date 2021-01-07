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

def calc_rouge_1_sort_F():
    F_rev = []
    for i in range(rev_num):
        # com_num = get_com_num(gen_revs_GTR[i], revs[i])
        # R = com_num / len(revs[i]) if len(revs[i]) else 0
        # P = com_num / len(gen_revs_GTR[i]) if len(gen_revs_GTR[i]) else 0

        com_num = get_com_num(gen_revs_GTR_add[i], revs[i])
        R = com_num / len(revs[i]) if len(revs[i]) else 0
        P = com_num / len(gen_revs_GTR_add[i]) if len(gen_revs_GTR_add[i]) else 0
        if R == 0 and P == 0:
            F1 = 0
        else:
            F1 = 2 * (R * P) / (R + P)
        F_rev.append([[F1],
                      [rate_real[i], revs[i]],
                      [pre_ratings_GTR[i], gen_revs_GTR[i]],
                      [pre_ratings_GTR_add[i], gen_revs_GTR_add[i]],
                      # [pre_ratings_NRT[i][0], gen_revs_NRT[i]],
                      # [pre_ratings_NM[i][0], gen_revs_NM[i]],
                      ])
    return sorted(F_rev, key=lambda x:x[0][0], reverse=True)[top_start:top_end]

def index_to_word(R_rev, para_trans_file):
    pkl_file = open(para_trans_file, 'rb')
    para = pickle.load(pkl_file)
    vocab = para['vocab']
    vocab = {v: k for k, v in vocab.items()}
    R_rev_word = []
    for m in R_rev:
        new_m = [m[0]]
        for line in m[1:]:
            rev = ""
            for index in line[1]:
                if index in vocab:
                    rev = rev + vocab[index] + " "
            new_m.append([line[0], rev])
        R_rev_word.append(new_m)
    return R_rev_word

def  read_files():
    with open(val_data_file, 'rb') as pkl_file:
        val_data = pickle.load(pkl_file)

    with open(gen_revs_file_GTR, 'rb') as pkl_file:
        gen_revs_GTR = pickle.load(pkl_file)
    with open(pre_raings_file_GTR, 'rb') as pkl_file:
        pre_ratings_GTR = pickle.load(pkl_file)

    with open(gen_revs_file_GTR_add, 'rb') as pkl_file:
        gen_revs_GTR_add = pickle.load(pkl_file)
    with open(pre_rating_file_GTR_add, 'rb') as pkl_file:
        pre_ratings_GTR_add = pickle.load(pkl_file)

    with open(gen_revs_file_NRT, 'rb') as pkl_file:
        gen_revs_NRT = pickle.load(pkl_file)
    with open(pre_raings_file_NRT, 'rb') as pkl_file:
        pre_ratings_NRT = pickle.load(pkl_file)

    with open(gen_revs_file_NM, 'rb') as pkl_file:
        gen_revs_NM = pickle.load(pkl_file)
    with open(pre_raings_file_NM, 'rb') as pkl_file:
        pre_ratings_NM = pickle.load(pkl_file)

    return val_data, remove_pad_word(gen_revs_GTR), pre_ratings_GTR, remove_pad_word(gen_revs_NRT), pre_ratings_NRT, \
           remove_pad_word(gen_revs_NM), pre_ratings_NM, gen_revs_GTR_add, pre_ratings_GTR_add

dataset = "Patio_Lawn_and_Garden"
TPS_DIR = '../common_data/' + dataset + '/'
val_data_file = TPS_DIR + "/val_trans_sl40_mf10_nodrop"
para_trans_file = TPS_DIR + "/para_trans_sl40_mf10_nodrop"

gen_revs_file_GTR = TPS_DIR + "gen_revs_nodrop.txt"
gen_revs_file_GTR_add = TPS_DIR + "gen_revs_GTR_KL0_no_pad_nodrop.txt"
pre_rating_file_GTR_add = TPS_DIR + "pre_raings_GTR_KL0_no_pad_nodrop.txt"

pre_raings_file_GTR = TPS_DIR + "pre_raings.txt"
gen_revs_file_NRT = TPS_DIR + "NRT_gen_revs_nodrop.txt"
pre_raings_file_NRT = TPS_DIR + "NRT_pre_ratings.txt"
gen_revs_file_NM = TPS_DIR + "NM_gen_revs.txt"
pre_raings_file_NM = TPS_DIR + "NM_pre_ratings.txt"

batch_size = 64

val_data, gen_revs_GTR, pre_ratings_GTR, gen_revs_NRT, pre_ratings_NRT, gen_revs_NM, pre_ratings_NM, \
            gen_revs_GTR_add, pre_ratings_GTR_add = read_files()

val_batchs = int(len(val_data)/batch_size)
val_data = val_data[:val_batchs*batch_size]

uids, iids, rate_real, revs = zip(*val_data)

rev_num = len(revs)
rev_len = len(revs[0])

#rate_pre = [0]*rev_num

revs = remove_pad_word(revs)

top_start = 0
top_end = 100
R_rev_index = calc_rouge_1_sort_F()
R_rev_word = index_to_word(R_rev_index, para_trans_file)
for m in R_rev_word:
    for line in m:
        print(line)
    print()
