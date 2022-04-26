import pickle

from ConvNet import ConvNet
import torch
import numpy as np
# swi = {'et': 0, 'pt': 1, 'tt': 2, 'cy': 3, 'ar': 4, 'ca': 5, 'de': 6, 'es': 7, 'eu': 8, 'en': 9, 'fr': 10, 'eo': 11, 'it': 12, 'kab': 13, 'rw': 14, 'ru': 15, 'zh-CN': 16, 'lv': 17, 'id': 18, 'hsb': 19, 'sl': 20, 'ta': 21, 'rm-sursilv': 22, 'el': 23, 'hu': 24, 'mn': 25, 'th': 26, 'sah': 27, 'fy-NL': 28, 'fa': 29}
rep = {0: 'et', 1: 'pt', 2: 'tt', 3: 'cy', 4: 'ar', 5: 'ca', 6: 'de', 7: 'es', 8: 'eu', 9: 'en', 10: 'fr', 11: 'eo', 12: 'it', 13: 'kab', 14: 'rw', 15: 'ru', 16: 'zh-CN', 17: 'lv', 18: 'id', 19: 'hsb', 20: 'sl', 21: 'ta', 22: 'rm-sursilv', 23: 'el', 24: 'hu', 25: 'mn', 26: 'th', 27: 'sah', 28: 'fy-NL', 29: 'fa'}


NUM_LANGUAGE = 30
path_to_sample = 'data/sub_pickles/test/en_100.pkl'
TRAINED_MODEL_PATH = 'trained_models/convN/13_4_2022_22_13_00/ConvNet_dialect_w-epoch_10.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    model = ConvNet(NUM_LANGUAGE).to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu')))
    return model


def get_packs_tags():
    with open(path_to_sample, 'rb') as f:
        pack = pickle.load(f)
        return pack[0]



def get_res(proba_pred_y):
    res =[]
    t = proba_pred_y[0]
    for i in t:
        x = np.exp(i.item())
        res.append(x)
    return res

def get_pred_y(proba_pred_y):
    res = get_res(proba_pred_y)
    max_res = []
    for sample in res:
        max_res.append(np.argmax(sample))
    return max_res


def get_Y(sample):
    model = get_model()
    with torch.no_grad():
        model.eval()
        proba_pred_y = model(sample.to(device))
        res = get_res(proba_pred_y)
        return res





    # with torch.no_grad():
    #     model.eval()
        # Y = model(sample.to(device))






def get_top_k_res(res, k):
    arr = np.array(res)
    top_k = arr.argsort(axis=0)[-k:]
    arr2 = []
    for j in reversed(range(len(top_k))):
        m = top_k[j]
        arr2.append((rep[m], round(arr[m], 4)))
    return arr2





# def get_top_k_res(res, k):
#     new_mat = []
#     for i in range(len(mat_res)):
#         arr = np.array(mat_res[i])
#         top_k = arr.argsort(axis=0)[-k:]
#         arr2 = []
#         for j in reversed(range(len(top_k))):
#             m = top_k[j]
#             arr2.append((rep[m],round(arr[m], 4)))
#         new_mat.append(arr2)
#     return new_mat


def get_res_in_k(top_k):
    sum_of_top_k = sum([i[1] for i in top_k])
    res_in_k = []
    for dup in top_k:
        res_in_k.append((dup[0],dup[1]/sum_of_top_k))
    return res_in_k

def get_language_by_vector(sample):
    list_of_x = [sample]
    x_stack = torch.stack(list_of_x)
    res = get_Y(x_stack )
    top_k = get_top_k_res(res, 3)
    res_in_k = get_res_in_k(top_k)
    return top_k, res_in_k

def main():
    sample = get_packs_tags()
    top_k ,res_in_k= get_language_by_vector(sample)
    # print(top_k)
    # print(res_in_k)
    ans = {}
    for i in range(len(top_k)):
        ans[top_k[i][0]] = (top_k[i][1], res_in_k[i][1])
    str_of_res = []
    for k, v in ans.items():
        str_of_res.append(f'For the {k} language the result is {v[0]} out of all and {v[1]} out of the top_3')
    final_answer = "\n".join(str_of_res)
    print(final_answer)


if __name__ == '__main__':
    main()




