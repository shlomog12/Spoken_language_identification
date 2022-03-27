import os
from datetime import datetime

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score

import math_func as f
from Data import Data
from cnn_model_definition import Convolutional_Language_Identification






TRAINED_MODEL_PATH = 'trained_models/Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_101.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Convolutional_Language_Identification().to(device)
model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
# model = torch.load(TRAINED_MODEL_PATH, map_location='cpu')

final_accuracy = np.array([0, 0, 0], dtype=float)
count_test = 0
ROOT_PATH = ''
ALL_DATA_PATH = ROOT_PATH+'data/pickles/total_data.pkl'
PATH_DATA_3 = ROOT_PATH +'data/pickles/db_3_langs.pkl'
TEST_RESULTS_PATH = 'results/only_test/'
now_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
dir_of_res_path = TEST_RESULTS_PATH + now_time   # same to loger
if not os.path.isdir(dir_of_res_path):
    os.makedirs(dir_of_res_path)
data = Data(ALL_DATA_PATH)
x_test = data.get_x_test()
y_test = data.get_y_test()
size_of_test = len(y_test)
batch_size = 16





with torch.no_grad():
    model.eval()
    test_epoch_idx = np.random.permutation(size_of_test)
    for b in range(int(np.ceil(size_of_test/batch_size))):
        test_batch_loc = test_epoch_idx[(b * batch_size):((b + 1) * batch_size)]
        mini_x_test, mini_y_test = x_test[test_batch_loc], y_test[test_batch_loc]
        proba_pred_y = model(mini_x_test.to(device))
        list_of_pred_y = f.get_pred_y(proba_pred_y)
        count_test +=1
        accuracy_list = []
        for k in [1, 5, 10]:
            accuracy_list += [f.top_k_accuracy(k, proba_pred_y, mini_y_test)]
        final_accuracy += np.array(accuracy_list)
    final_accuracy /= count_test
    # f1_score = f1_score(mini_y_test, proba_pred_y)
    # print(f1_score)
new_final_accuracy = [round(x * 100, 3) for x in list(final_accuracy)]
results_df = pd.DataFrame([], columns=['top_1_test_acc', 'top_5_test_acc', 'top_10_test_acc'])
results_df.loc[len(results_df)] = new_final_accuracy
results_df.to_excel(dir_of_res_path + '/final_report.xlsx')

