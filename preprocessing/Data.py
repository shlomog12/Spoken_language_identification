import torch
import glob
import pickle
import os
from sklearn.model_selection import train_test_split


class Data:

    def __init__(self):
        self.init_define()

    def init_define(self):
        self.path_to_data = '../data'
        # self.path_to_data = '/content/drive/MyDrive/Spoken-language-identification/data'
        self.train_path = f'{self.path_to_data}/sub_pickles/train/'
        self.test_path = f'{self.path_to_data}/sub_pickles/test/'
        self.limit_train = 5000
        self.limit_test = 1250
        self.NUM_LANGUAGES = 49
        self.switcher = {}


    def init_data_test(self):
        pks_of_test = self.get_packs_tags(self.test_path)
        X_test, Y_test, _, amounts_test = self.unit_packs(pks_of_test, self.limit_test)
        print(f' len(X_test) = {len(X_test)}, len(Y_test)= {len(Y_test)} ')
        print(self.switcher)
        self.X_test, self.Y_test = self.to_tensors(X_test, Y_test)


    def init_data_train(self):
        pks_of_train = self.get_packs_tags(self.train_path)
        X_train, Y_train, weights, amounts_train = self.unit_packs(pks_of_train, self.limit_train)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, shuffle=True)
        print(f' len(X_train) = {len(X_train)},  len(Y_train) = {len(Y_train)}')
        print(f' len(X_val) = {len(X_val)}, len(Y_val)= {len(Y_val)} ')
        # print(f' len(X_test) = {len(X_test)}, len(Y_test)= {len(Y_test)} ')
        print(self.switcher)
        # to tensor
        self.X_train, self.Y_train = self.to_tensors(X_train, Y_train)
        self.X_val, self.Y_val = self.to_tensors(X_val, Y_val)
        self.weights = torch.tensor(weights)
        print('after to tensor')

    def get_num_by_tag(self, tag):
        self.switcher[tag] = self.switcher.get(tag, len(self.switcher))
        return self.switcher[tag]

    def get_packs_tags(self, path):
        packs = []
        for file in glob.glob(f'{path}*.pkl'):
            fname = os.path.basename(file)
            tag = fname.split('_')[0]
            path_to_current = path + fname
            with open(path_to_current, 'rb') as f:
                pack = pickle.load(f)
                packs.append((pack, tag))
        return packs

    def unit_packs(self, packs, limit):
        amounts = [0] * self.NUM_LANGUAGES
        X = []
        Y = []
        for pack in packs:
            tensors, tag = pack[0], pack[1]
            num = self.get_num_by_tag(tag)
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
        return X, Y, weights, amounts

    def to_tensors(self, X, Y):
        return torch.stack(X), torch.tensor(Y)

    def get_x_train(self):
        return self.X_train

    def get_y_train(self):
        return self.Y_train

    def get_x_val(self):
        return self.X_val

    def get_y_val(self):
        return self.Y_val

    def get_weights(self):
        return self.weights

    def get_data(self):
        return self.X_train, self.Y_train, self.X_val, self.Y_val, self.weights

    def get_x_test(self):
        return self.X_test

    def get_y_test(self):
        return self.Y_test