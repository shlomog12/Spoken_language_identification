
import numpy as np


def top_k_accuracy(k, proba_pred_y, mini_y_test):
    top_k_pred = proba_pred_y.argsort(axis=1)[:, -k:]
    final_pred = [False] * len(mini_y_test)
    for j in range(len(mini_y_test)):
        final_pred[j] = mini_y_test[j] in top_k_pred
    return np.mean(final_pred)

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