
import numpy as np


def add(x,y):
    for i in range(len(x)):
        x[i] +=y[i]
    return x


def add(x,y):
    for i in range(len(x)):
        x[i] +=y[i]
    return x
#

# print(mat_res)
#
# # mat_res
def normal_mat(mat):
    for i in range(len(mat)):
        mat[i] = normal_arr(mat[i])
    return mat


def normal_arr(arr):
    sum = np.sum(np.array(arr))
    if sum == 0:
        return arr
    for i in range(len(arr)):
        arr[i] = arr[i] / sum
        arr[i] = round(arr[i], 4)
    return arr

def top_k_accuracy(k, proba_pred_y, mini_y_test):
    top_k_pred = proba_pred_y.argsort(axis=1)[:, -k:]
    final_pred = [False] * len(mini_y_test)
    for j in range(len(mini_y_test)):
        final_pred[j] = mini_y_test[j] in top_k_pred
    return np.mean(final_pred)

def update_mat(mat ,proba_pred_y, mini_y_test):
    xx = get_res(proba_pred_y)
    for i in range(len(mini_y_test)):
        t = mini_y_test[i]
        mat[t] = add(mat[t], xx[i])
    return mat


def get_res(proba_pred_y):
    res =[]
    for t in proba_pred_y:
        resI =[]
        for i in t:
            x = np.exp(i.item())
            resI.append(x)
        res.append(resI)
    return res

def get_pred_y(proba_pred_y):
    res = get_res(proba_pred_y)
    max_res = []
    for sample in res:
        max_res.append(np.argmax(sample))
    return max_res

def get_top_k_res(mat_res,k ,rep):
    new_mat = []
    for i in range(len(mat_res)):
        arr = np.array(mat_res[i])
        top_k = arr.argsort(axis=0)[-k:]
        arr2 = []
        for j in reversed(range(len(top_k))):
            m = top_k[j]
            arr2.append((rep[m],round(arr[m], 4)))
        new_mat.append(arr2)
    return new_mat