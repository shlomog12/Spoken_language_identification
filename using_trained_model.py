import os
from datetime import datetime
import pytz
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score

import math_func as f
from ConvNet import ConvNet
from preprocessing.Data import Data
from cnn_model_definition import Convolutional_Language_Identification

data = Data()
data.init_data_test()
x_test = data.get_x_test()
y_test = data.get_y_test()
size_of_test = len(y_test)

TRAINED_MODEL_PATH = 'trained_models/aaaa/Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_10.pth'
print(TRAINED_MODEL_PATH )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ConvNet(data.get_num_language()).to(device)
model = Convolutional_Language_Identification(data.get_num_language()).to(device)
model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu')))
# model = torch.load(TRAINED_MODEL_PATH, map_location='cpu')

final_accuracy = np.array([0, 0, 0, 0], dtype=float)
final_f_score = np.array([0, 0], dtype=float)
ROOT_PATH = ''
ALL_DATA_PATH = ROOT_PATH+'data/pickles/total_data.pkl'
PATH_DATA_3 = ROOT_PATH +'data/pickles/db_3_langs.pkl'
TEST_RESULTS_PATH = 'results/only_test/'

timezone = pytz.timezone("Israel")
now_time = datetime.now(timezone).strftime("%d-%m-%Y_%H-%M-%S")

dir_of_res_path = f'{TEST_RESULTS_PATH}{now_time}_{TRAINED_MODEL_PATH[70:-4]}'
if not os.path.isdir(dir_of_res_path):
    os.makedirs(dir_of_res_path)

batch_size = 16

# size_l = 49
# size_l = 29
size_l = data.get_num_language()
mat_res = [[0 for i in range(size_l)] for j in range(size_l)]

def get_f_score(mini_y_test, preds):
    f_list = [f1_score(mini_y_test, preds, average="macro"), f1_score(mini_y_test, preds, average="micro")]
    return f_list


with torch.no_grad():
    model.eval()

    test_epoch_idx = np.random.permutation(size_of_test)
    num_test = int(np.ceil(size_of_test/batch_size))

    for b in tqdm(range(num_test)):
        test_batch_loc = test_epoch_idx[(b * batch_size):((b + 1) * batch_size)]
        mini_x_test, mini_y_test = x_test[test_batch_loc], y_test[test_batch_loc]
        proba_pred_y = model(mini_x_test.to(device))
        preds = f.get_pred_y(proba_pred_y)

        mat_res = f.update_mat(mat_res ,proba_pred_y, mini_y_test)

        f_score = get_f_score(mini_y_test, preds)
        accuracy_list = []
        for k in [1, 2, 3, 4]:
            accuracy_list += [f.top_k_accuracy(k, proba_pred_y, mini_y_test)]
        final_accuracy += np.array(accuracy_list)
        final_f_score += np.array(f_score)
    final_accuracy /= num_test
    final_f_score /=num_test
new_final_accuracy = [round(x * 100, 3) for x in list(final_accuracy)]
results_df = pd.DataFrame([], columns=['top_1_test_acc', 'top_2_test_acc', 'top_3_test_acc', 'top_4_test_acc', 'f_score_macro', 'f_score_micro'])
results_df.loc[len(results_df)] = new_final_accuracy + [final_f_score[0],final_f_score[1]]
results_df.to_excel(dir_of_res_path + '/final_report.xlsx')



# swi = {'et': 0, 'cs': 1, 'pt': 2, 'pl': 3, 'tt': 4, 'cy': 5, 'ar': 6, 'ca': 7, 'de': 8, 'es': 9, 'eu': 10, 'en': 11, 'fr': 12, 'eo': 13, 'it': 14, 'kab': 15, 'rw': 16, 'nl': 17, 'ru': 18, 'zh-CN': 19, 'br': 20, 'cv': 21, 'lt': 22, 'rm-vallader': 23, 'sv-SE': 24, 'lv': 25, 'lg': 26, 'id': 27, 'tr': 28, 'hsb': 29, 'ka': 30, 'sl': 31, 'ta': 32, 'ia': 33, 'zh-TW': 34, 'rm-sursilv': 35, 'mt': 36, 'el': 37, 'dv': 38, 'hu': 39, 'mn': 40, 'ro': 41, 'th': 42, 'sah': 43, 'ky': 44, 'zh-HK': 45, 'fy-NL': 46, 'uk': 47, 'fa': 48}
# swi = {'et': 0, 'cs': 1, 'pt': 2, 'pl': 3, 'tt': 4, 'cy': 5, 'ar': 6, 'ca': 7, 'de': 8, 'es': 9, 'eu': 10, 'en': 11, 'fr': 12, 'eo': 13, 'it': 14, 'kab': 15, 'rw': 16, 'nl': 17, 'ru': 18, 'zh-CN': 19, 'dv': 20, 'hu': 21, 'mn': 22, 'ro': 23, 'th': 24, 'zh-HK': 25, 'fy-NL': 26, 'uk': 27, 'fa': 28}
# swi = {'et': 0, 'pt': 1, 'tt': 2, 'cy': 3, 'ar': 4, 'ca': 5, 'de': 6, 'es': 7, 'eu': 8, 'en': 9, 'fr': 10, 'eo': 11, 'it': 12, 'kab': 13, 'rw': 14, 'ru': 15, 'zh-CN': 16, 'lv': 17, 'id': 18, 'hsb': 19, 'sl': 20, 'ta': 21, 'rm-sursilv': 22, 'el': 23, 'hu': 24, 'mn': 25, 'th': 26, 'sah': 27, 'fy-NL': 28, 'fa': 29}
swi = data.get_switcher()
rep = {}
for k, v in swi.items():
    rep[v] = k
col_name =["real \ guess"] + list(swi.keys())
mat_res_df = pd.DataFrame([], columns=col_name)
mat_res = f.normal_mat(mat_res)
for i in range(len(mat_res)):
    mat_res_df.loc[len(mat_res_df)] = [rep[i]] + mat_res[i]
mat_res_df.to_excel(dir_of_res_path + '/mat_res.xlsx')


k_to_top = 5
top_k = f.get_top_k_res(mat_res, k_to_top, rep)
top_k_df = pd.DataFrame([], columns=range(1, k_to_top+2))
for i in range(len(top_k)):
    top_k_df.loc[len(top_k_df)] = [rep[i]] + top_k[i]
top_k_df.to_excel(dir_of_res_path + '/top_5_df.xlsx')