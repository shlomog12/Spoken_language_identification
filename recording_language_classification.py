from cnn_model_definition import Convolutional_Language_Identification
import torch
import numpy as np
import math_func as f


NUM_LANGUAGE = 30
TRAINED_MODEL_PATH = 'trained_models/aaaa/Convolutional_Speaker_Identification_Log_Softmax_Model-epoch_10.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    model = Convolutional_Language_Identification(NUM_LANGUAGE).to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu'))
    return model


def get_res(proba_pred_y):
    res =[]
    for t in proba_pred_y:
        resI =[]
        for i in t:
            x = np.exp(i.item())
            resI.append(x)
        res.append(resI)
    return res


def get_Y(sample):
    mat_res = [0 for i in range(NUM_LANGUAGE)]
    model = get_model()
    with torch.no_grad():
        model.eval()

        mini_x_test = sample
        proba_pred_y = model(sample.to(device))



        mat_res = f.update_mat(mat_res ,proba_pred_y, mini_y_test)

        accuracy_list = []
        for k in [1, 2, 3 4],:
            accuracy_list += [f.top_k_accuracy(k, proba_pred_y, mini_y_test)]




    # with torch.no_grad():
    #     model.eval()
        # Y = model(sample.to(device))

















def get_language_by_vector(sample):