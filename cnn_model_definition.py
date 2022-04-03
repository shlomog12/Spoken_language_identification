"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn
NUM_OF_LANGUAGES = 49
import torch

DROP_OUT = 0.5
DIMENSION = 512 * 300


class Convolutional_Language_Identification(nn.Module):

    def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
        return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2

    def __init__(self):

        super().__init__()



        # karnel this window in conv that all m[i][j] = sum(m[i-1][j]*karnel[i-1][j],m[i+1][j]*karnel[i+1][j] ,,,,,,,) all window
        # padding all side for karnel work on frame
        #                         (in,out) -> ?
        self.conv_2d_1 = nn.Conv2d(1, 96, kernel_size=(3, 3), stride=(2, 2), padding=1)
        # nermul
        self.bn_1 = nn.BatchNorm2d(96)
        # all x = max in window  -> m[i][j] = max(m[i-1][j],m[i+1][j] ,,,,,,,)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_2d_2 = nn.Conv2d(96, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn_2 = nn.BatchNorm2d(256)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.conv_2d_3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.bn_3 = nn.BatchNorm2d(384)

        self.conv_2d_4 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.bn_4 = nn.BatchNorm2d(256)

        self.conv_2d_5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 2) ,padding=1)

        self.conv_2d_6 = nn.Conv2d(256, 4096, kernel_size=(3, 3), padding=1)
        self.drop_1 = nn.Dropout(p=DROP_OUT)

        self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_1 = nn.Linear(4096, 1024)
        self.drop_2 = nn.Dropout(p=DROP_OUT)

        self.dense_2 = nn.Linear(1024, NUM_OF_LANGUAGES)

    def forward(self, X):

        x = nn.ReLU()(self.conv_2d_1(X))
        x = self.bn_1(x)
        x = self.max_pool_2d_1(x)

        x = nn.ReLU()(self.conv_2d_2(x))
        x = self.bn_2(x)
        x = self.max_pool_2d_2(x)

        x = nn.ReLU()(self.conv_2d_3(x))
        x = self.bn_3(x)

        x = nn.ReLU()(self.conv_2d_4(x))
        x = self.bn_4(x)

        x = nn.ReLU()(self.conv_2d_5(x))
        x = self.bn_5(x)
        x = self.max_pool_2d_3(x)

        x = nn.ReLU()(self.conv_2d_6(x))
        x = self.drop_1(x)
        x = self.global_avg_pooling_2d(x)

        x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer
        x = nn.ReLU()(self.dense_1(x))
        x = self.drop_2(x)

        x = self.dense_2(x)
        y = nn.LogSoftmax(dim=1)(x)   # consider using Log-Softmax

        return y

    # number of iteration on all project
    def get_epochs(self):
        return 500

    def get_learning_rate(self):
        return 0.0001

    # in all iteration take 16 random Recordings
    # num iteration = size(data)/bath_size
    def get_batch_size(self):
        return 16

    def to_string(self):
        return "Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_"

