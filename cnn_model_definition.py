
from torch import nn
# NUM_OF_LANGUAGES = 49
# NUM_OF_LANGUAGES = 29
# NUM_OF_LANGUAGES = 30
import torch

DROP_OUT = 0.5
DIMENSION = 512 * 300
DROP_OUT = 0.5


# class ConvNet(nn.Module):
class Convolutional_Language_Identification(nn.Module):

    def __init__(self, num_of_classes):
        super().__init__()
        # Hyper parameters
        self.epochs = 45
        self.batch_size = 44
        self.learning_rate = 0.0001
        # self.dataset = dataset
        # Model Architecture
        self.first_conv = nn.Conv2d(1, 96, kernel_size=(5, 5), padding=1)
        self.first_bn = nn.BatchNorm2d(96)
        self.first_polling = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.second_conv = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=1)
        self.second_bn = nn.BatchNorm2d(256)
        self.second_polling = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))

        self.third_conv = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.third_bn = nn.BatchNorm2d(384)

        self.forth_conv = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.forth_bn = nn.BatchNorm2d(256)

        self.fifth_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.fifth_bn = nn.BatchNorm2d(256)
        self.fifth_polling = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))

        self.sixth_conv = nn.Conv2d(256, 64, kernel_size=(2, 2), padding=1)
        self.first_drop = nn.Dropout(p=DROP_OUT)

        self.avg_polling = nn.AdaptiveAvgPool2d((1, 1))
        self.first_dense = nn.Linear(64, 1024)
        self.second_drop = nn.Dropout(p=DROP_OUT)

        self.second_dense = nn.Linear(1024, num_of_classes)

    def forward(self, X):
        x = nn.ReLU()(self.first_conv(X))
        x = self.first_bn(x)
        x = self.first_polling(x)

        x = nn.ReLU()(self.second_conv(x))
        x = self.second_bn(x)
        x = self.second_polling(x)

        x = nn.ReLU()(self.third_conv(x))
        x = self.third_bn(x)

        x = nn.ReLU()(self.forth_conv(x))
        x = self.forth_bn(x)

        x = nn.ReLU()(self.fifth_conv(x))
        x = self.fifth_bn(x)
        x = self.fifth_polling(x)

        x = nn.ReLU()(self.sixth_conv(x))
        x = self.first_drop(x)
        x = self.avg_polling(x)

        x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer

        x = nn.ReLU()(self.first_dense(x))
        x = self.second_drop(x)

        x = self.second_dense(x)
        y = nn.LogSoftmax(dim=1)(x)  # consider using Log-Softmax

        return y

    def to_string(self):
        return "train_dialect_with_w_dolev44_-epoch_"

    def get_epochs(self):
        return self.epochs

    def get_learning_rate(self):
        return self.learning_rate

    def get_batch_size(self):
        return self.batch_size























# *****************************************************************************************************************************







# """******************************************************************
# The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
# ******************************************************************"""
#
# from torch import nn
# # NUM_OF_LANGUAGES = 49
# # NUM_OF_LANGUAGES = 29
# # NUM_OF_LANGUAGES = 30
# import torch
#
# DROP_OUT = 0.5
# DIMENSION = 512 * 300
#
#
# class Convolutional_Language_Identification(nn.Module):
#
#     def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
#         return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2
#
#     def __init__(self, num_language):
#         super().__init__()
#
#
#
#         # karnel this window in conv that all m[i][j] = sum(m[i-1][j]*karnel[i-1][j],m[i+1][j]*karnel[i+1][j] ,,,,,,,) all window
#         # padding all side for karnel work on frame
#         #                         (in,out) -> ?
#         self.conv_2d_1 = nn.Conv2d(1, 96, kernel_size=(3, 3), stride=(2, 2), padding=1)
#         # nermul
#         self.bn_1 = nn.BatchNorm2d(96)
#         # all x = max in window  -> m[i][j] = max(m[i-1][j],m[i+1][j] ,,,,,,,)
#         self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
#
#         self.conv_2d_2 = nn.Conv2d(96, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.bn_2 = nn.BatchNorm2d(256)
#         self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
#
#         self.conv_2d_3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
#         self.bn_3 = nn.BatchNorm2d(384)
#
#         self.conv_2d_4 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
#         self.bn_4 = nn.BatchNorm2d(256)
#
#         self.conv_2d_5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
#         self.bn_5 = nn.BatchNorm2d(256)
#         self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 2) ,padding=1)
#
#         self.conv_2d_6 = nn.Conv2d(256, 4096, kernel_size=(3, 3), padding=1)
#         self.drop_1 = nn.Dropout(p=DROP_OUT)
#
#         self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
#         self.dense_1 = nn.Linear(4096, 1024)
#         self.drop_2 = nn.Dropout(p=DROP_OUT)
#
#         self.dense_2 = nn.Linear(1024, num_language)
#
#     def forward(self, X):
#
#         x = nn.ReLU()(self.conv_2d_1(X))
#         x = self.bn_1(x)
#         x = self.max_pool_2d_1(x)
#
#         x = nn.ReLU()(self.conv_2d_2(x))
#         x = self.bn_2(x)
#         x = self.max_pool_2d_2(x)
#
#         x = nn.ReLU()(self.conv_2d_3(x))
#         x = self.bn_3(x)
#
#         x = nn.ReLU()(self.conv_2d_4(x))
#         x = self.bn_4(x)
#
#         x = nn.ReLU()(self.conv_2d_5(x))
#         x = self.bn_5(x)
#         x = self.max_pool_2d_3(x)
#
#         x = nn.ReLU()(self.conv_2d_6(x))
#         x = self.drop_1(x)
#         x = self.global_avg_pooling_2d(x)
#
#         x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer
#         x = nn.ReLU()(self.dense_1(x))
#         x = self.drop_2(x)
#
#         x = self.dense_2(x)
#         y = nn.LogSoftmax(dim=1)(x)   # consider using Log-Softmax
#
#         return y
#
#     # number of iteration on all project
#     def get_epochs(self):
#         return 500
#
#     def get_learning_rate(self):
#         return 0.0001
#
#     # in all iteration take 16 random Recordings
#     # num iteration = size(data)/bath_size
#     def get_batch_size(self):
#         return 16
#
#     def to_string(self):
#         return "Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_"



# gilad
# """******************************************************************
# The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
# ******************************************************************"""
#
# from torch import nn
# # NUM_OF_LANGUAGES = 49
# # NUM_OF_LANGUAGES = 29
# import torch
#
# DROP_OUT = 0.5
# DIMENSION = 512 * 300
#
#
# class Convolutional_Language_Identification(nn.Module):
#
#     def cal_paddind_shape(self, new_shape, old_shape, kernel_size, stride_size):
#         return (stride_size * (new_shape - 1) + kernel_size - old_shape) / 2
#
#     def __init__(self, num_language):
#         super().__init__()
#
#         self.first_conv = nn.Conv2d(1, 96, kernel_size=(5, 5), padding=1)  # (96, 147, 30)
#         self.first_bn = nn.BatchNorm2d(96)
#         self.first_polling = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))  # (96, 73, 14)
#
#         self.second_conv = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=1)  # (256, 71, 12)
#         self.second_bn = nn.BatchNorm2d(256)
#         self.second_polling = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))  # (256, 69, 10)
#
#         self.third_conv = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)  # (384, 69, 10 )
#         self.third_bn = nn.BatchNorm2d(384)
#
#         self.forth_conv = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)  # (256, 69, 10)
#         self.forth_bn = nn.BatchNorm2d(256)
#
#         self.fifth_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)  # (256, 69, 10)
#         self.fifth_bn = nn.BatchNorm2d(256)
#         self.fifth_polling = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))  # (256, 68, 9)
#
#         self.sixth_conv = nn.Conv2d(256, 64, kernel_size=(2, 2), padding=1)  # (64, 69, 10)
#
#         self.seventh_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)  # (64, 69, 10)
#         self.seventh_polling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # (64, 34, 5)
#
#         self.eighth_conv = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1)  # (32, 34, 5)
#         self.first_drop = nn.Dropout(p=DROP_OUT)
#
#         self.avg_polling = nn.AdaptiveAvgPool2d((1, 1))
#         self.first_dense = nn.Linear(32, 1024)
#
#         self.second_drop = nn.Dropout(p=DROP_OUT)
#
#         self.second_dense = nn.Linear(1024, 512)
#
#         self.third_drop = nn.Dropout(p=DROP_OUT)
#         self.third_dense = nn.Linear(512, num_language)
#
#     def forward(self, X):
#         x = nn.ReLU()(self.first_conv(X))
#         x = self.first_bn(x)
#         x = self.first_polling(x)
#
#         x = nn.ReLU()(self.second_conv(x))
#         x = self.second_bn(x)
#         x = self.second_polling(x)
#
#         x = nn.ReLU()(self.third_conv(x))
#         x = self.third_bn(x)
#
#         x = nn.ReLU()(self.forth_conv(x))
#         x = self.forth_bn(x)
#
#         x = nn.ReLU()(self.fifth_conv(x))
#         x = self.fifth_bn(x)
#         x = self.fifth_polling(x)
#
#         x = nn.ReLU()(self.sixth_conv(x))
#
#         x = nn.ReLU()(self.seventh_conv(x))
#         x = self.seventh_polling(x)
#
#         x = nn.ReLU()(self.eighth_conv(x))
#
#         x = self.first_drop(x)
#         x = self.avg_polling(x)
#
#         x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer
#
#         x = nn.ReLU()(self.first_dense(x))
#         x = self.second_drop(x)
#
#         x = nn.ReLU()(self.second_dense(x))
#         x = self.third_drop(x)
#         x = self.third_dense(x)
#         y = nn.LogSoftmax(dim=1)(x)  # consider using Log-Softmax
#
#         return y
#
#     # number of iteration on all project
#     def get_epochs(self):
#         return 150
#
#     def get_learning_rate(self):
#         return 0.0001
#
#     # in all iteration take 16 random Recordings
#     # num iteration = size(data)/bath_size
#     def get_batch_size(self):
#         return 34
#
#     def to_string(self):
#         return "Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_"