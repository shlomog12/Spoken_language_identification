import numpy as np
import glob
import pickle
import os
from sklearn.model_selection import train_test_split
train_packs = []
test_packs = []
NUM_LANGUAGES = 49


path_to_data = '../data'
# path_to_data = '/content/drive/MyDrive/Spoken-language-identification/data'

train_path = f'{path_to_data}/sub_pickles/train/'
test_path = f'{path_to_data}/sub_pickles/test/'
limit_train = 5000
limit_test = 1250
switcher = {}
# prev = 'br'


def get_num_by_tag(tag):
    switcher[tag] = switcher.get(tag, len(switcher))
    return switcher[tag]

def get_packs_tags(path):
    packs = []
    for file in glob.glob(f'{path}*.pkl'):
        fname = os.path.basename(file)
        tag = fname.split('_')[0]
        path_to_current = path + fname

        with open(path_to_current, 'rb') as f:
            pack = pickle.load(f)
            packs.append((pack, tag))
    return packs

def unit_packs(packs, limit):
    amounts = [0] * NUM_LANGUAGES
    X = []
    Y = []
    for pack in packs:
        tensors, tag = pack[0], pack[1]
        num = get_num_by_tag(tag)
        if amounts[num] == limit:
            continue
        for tensor in tensors:
            if amounts[num] == limit:
                break
            X.append(tensor)
            amounts[num] += 1
            Y.append(num)

    max_amount = max(amounts)
    weights = [max_amount / amount for amount in amounts]
    return X, Y, weights , amounts


pks_of_train = get_packs_tags(train_path)
pks_of_test = get_packs_tags(test_path)

X_train, Y_train, weights , amounts_train = unit_packs(pks_of_train, limit_train)
X_test, Y_test, _ , amounts_test = unit_packs(pks_of_test, limit_test)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, shuffle=True)

print(f' len(X_train) = {len(X_train)},  len(Y_train) = {len(Y_train)}')
print(f' len(X_val) = {len(X_val)}, len(Y_val)= {len(Y_val)} ')
print(f' len(X_test) = {len(X_test)}, len(Y_test)= {len(Y_test)} ')


data = [X_train, Y_train, weights, X_val, Y_val, X_test, Y_test]
path = f'{path_to_data}/pickles/total_big.pkl'
with open(path, 'wb') as f:
    pickle.dump(data, f)








