import pickle
import torch





class Data:


    def __init__(self,  path=''):
        self.switcher = {}
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for i in range(len(data)):
                if i%2 ==0:
                    data[i] = torch.stack(data[i])
                else:
                    data[i] = self.getTensor(data[i])
            self.data = data
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test =data


    def convert_to_int(self, list_y):
        self.init_switcher(list_y)
        for i in range(len(list_y)):
            list_y[i] = self.switcher.get(list_y[i])
        return list_y

    def init_switcher(self, list_y):
        if self.switcher != {}:
            return
        list_xx = list(set(list_y))
        list_xx.sort()
        for i in range(len(list_xx)):
            self.switcher[list_xx[i]] = i

    def getTensor(self ,list_y):
        list_y = self.convert_to_int(list_y)
        return torch.tensor(list_y)



    def get_x_train(self):
        return self.x_train

    def get_y_train(self):
        return self.y_train

    def get_x_val(self):
        return self.x_val

    def get_y_val(self):
        return self.y_val

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test

    def get_data(self):
        return self.data