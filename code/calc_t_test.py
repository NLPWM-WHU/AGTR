import numpy as np
import pickle
from scipy import stats

dataset = "Patio_Lawn_and_Garden"
TPS_DIR = '../common_data/' + dataset + '/'
file_ALFM_pre_rate = TPS_DIR + "Automotive2.predict"
file_GTR_pre_rate = TPS_DIR + "pre_raings_GTR_KL1.txt"
file_GTR_pre = TPS_DIR + "pre_raings.txt"
file_NRT_pre = TPS_DIR + "NRT_pre_ratings.txt"
file_NM_pre = TPS_DIR + "NM_pre_ratings.txt"

# ALFM_pre_rate = []
# with open(file_ALFM_pre_rate, 'r') as pkl_file:
#     for line in pkl_file:
#         rate = float(line.split(",")[-1])
#         ALFM_pre_rate.append(rate)
#
# with open(file_GTR_pre_rate, 'rb') as pkl_file:
#     GTR_pre_rate = pickle.load(pkl_file)
#
# data_num = len(GTR_pre_rate)
# ALFM_pre_rate = ALFM_pre_rate[:data_num]
#t = stats.ttest_ind(ALFM_pre_rate, GTR_pre_rate)

with open(file_GTR_pre, 'rb') as pkl_file:
    pre_ratings_GTR = pickle.load(pkl_file)
with open(file_NRT_pre, 'rb') as pkl_file:
    pre_ratings_NRT = pickle.load(pkl_file)
    pre_ratings_NRT = [r[0] for r in pre_ratings_NRT]
with open(file_NM_pre, 'rb') as pkl_file:
    pre_ratings_NM = pickle.load(pkl_file)
    pre_ratings_NM = [r[0] for r in pre_ratings_NM]
t_NRT = stats.ttest_ind(pre_ratings_NRT, pre_ratings_GTR)
t_NM = stats.ttest_ind(pre_ratings_NM, pre_ratings_GTR)
print("t_NRT", t_NRT)
print("t_NM", t_NM)
