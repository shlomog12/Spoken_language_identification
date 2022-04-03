import math_func as f
import numpy as np
import pandas as pd
# d = {}
# li = ['a','b','c','a','a','b','d']
# for x in li:
#     d[x] = d.get(x, 0) + 1
#
# print(d)
# switcher = {}
# def get_num_by_tag(tag):
#     switcher[tag] = switcher.get(tag, len(switcher))
#     return switcher[tag]
#
# li = ['a','b','c','a','a','b','d']
# nums = [0] * len(li)
# for x in range(len(li)):
#     nums[x] = get_num_by_tag(li[x])

# print(nums)

# for i in range(1,10+1):
#     print(i)

# for i in range(1000):
#     for j in range(100):
#         if j > 20:
#             break
#         print(f'{i}   - {j}')









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
    return arr

size_e = 5
mat_res = [[0 for i in range(size_e)] for j in range(size_e)]

mini_y_test = [1, 4, 2]
xx = [[0.2, 0.4, 0.1, 0.1, 0.2],[0.2, 0.1, 0.1, 0.2, 0.4],  [0.2, 0.1,0.4, 0.1, 0.2]]

for i in range(len(mini_y_test)):
    t = mini_y_test[i]
    mat_res[t] = add(mat_res[t], xx[i])
#
# print(mat_res)
for i in range(len(mini_y_test)):
    t = mini_y_test[i]
    mat_res[t] = add(mat_res[t], xx[i])
print(mat_res)
mat_res = normal_mat(mat_res)
print(mat_res)
# top_k_mat = f.get_top_k_res(mat_res, 3)
# print(top_k_mat)

# t = {'et': 0, 'cs': 1, 'pt': 2, 'pl': 3, 'tt': 4, 'cy': 5, 'ar': 6, 'ca': 7, 'de': 8, 'es': 9, 'eu': 10, 'en': 11, 'fr': 12, 'eo': 13, 'it': 14, 'kab': 15, 'rw': 16, 'nl': 17, 'ru': 18, 'zh-CN': 19, 'br': 20, 'cv': 21, 'lt': 22, 'rm-vallader': 23, 'sv-SE': 24, 'lv': 25, 'lg': 26, 'id': 27, 'tr': 28, 'hsb': 29, 'ka': 30, 'sl': 31, 'ta': 32, 'ia': 33, 'zh-TW': 34, 'rm-sursilv': 35, 'mt': 36, 'el': 37, 'dv': 38, 'hu': 39, 'mn': 40, 'ro': 41, 'th': 42, 'sah': 43, 'ky': 44, 'zh-HK': 45, 'fy-NL': 46, 'uk': 47, 'fa': 48}
# print(t.keys)
#
swi = {'et': 0, 'cs': 1, 'pt': 2, 'pl': 3, 'tt': 4}
rep = {}
for k,v in swi.items():
    rep[v] =k
# col_name =["real \ guess"] + list(swi.keys())
# mat_res_df = pd.DataFrame([], columns=col_name)
# for i in range(len(mat_res)):
    # mat_res_df.loc[len(mat_res_df)] = [rep[i]]+ mat_res[i]

k_to_top = 3
top_k = f.get_top_k_res(mat_res, k_to_top, rep)
top_k_df = pd.DataFrame([], columns=range(1, k_to_top+2))
for i in range(len(top_k)):
    top_k_df.loc[len(top_k_df)] = [rep[i]] + top_k[i]
top_k_df.to_excel('./top_5_test_df.xlsx')