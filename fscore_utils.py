import numpy as np
epsilon = 0.00000001
# only for example
def pr_ts(num_classes):
    return np.random.randint(low=0,high=num_classes,size=10)

def c_recall(pred,tst):
    tr_pos = c_tpositive(pred,tst)
    fl_neg = c_fnegative(pred,tst)
    return tr_pos/(tr_pos+fl_neg+epsilon)

def make_mask(arr,clss):
    k = [1 if arr[i] == clss else 0 for i in range(len(arr))]
    return np.asarray(k)


def c_fnegative(pred,tst):
    k = [1 if pred[i] == 0 and tst[i] == 1 else 0 for i in range(len(pred))]
    return np.sum(np.asarray(k))


def c_tpositive(pred,tst):
    k = [1 if pred[i] == 1 and tst[i] == 1 else 0 for i in range(len(pred))]
    return np.sum(np.asarray(k))


def c_fpositive(pred,tst):
    k = [1 if pred[i] == 1 and tst[i] == 0 else 0 for i in range(len(pred))]
    return np.sum(np.asarray(k))


def c_precision(pred,tst):
    tr_pos = c_tpositive(pred,tst)
    fl_pos = c_fpositive(pred,tst)
    return (tr_pos/(tr_pos+fl_pos+epsilon))


def c_fscore(prec,rec):
    return 2*(prec*rec/(prec+rec+epsilon))

def create_multi_accmat(pred,tst,num_cls):
    mat = []
    for i in range(num_cls):
        masked_pred = make_mask(pred,i)
        masked_tst = make_mask(tst,i)
        prec = c_precision(masked_pred,masked_tst)
        rec = c_recall(masked_pred,masked_tst)
        t_pos = c_tpositive(masked_pred, masked_tst)
        fscore = c_fscore(prec,rec)
        line = [fscore,t_pos,rec,prec]
        mat.append(line)
    return np.asarray(mat)


def c_micro_macro_fscore(pred,tst,num_cls):
    """
        Main function, computes micro and macro fscore
        :param pred: np array of predictions
        :param tst: np array of real y's
        :param num_cls: num of classes. note - the whole file assumes classe are in [0,num_cls-1]
        :return: micro fscore, macro fscore
        """
    mat = create_multi_accmat(pred,tst,num_cls)
    macro_fscore = np.sum(mat[:,0])/num_cls
    sum_tpos = np.sum(mat[:,1])
    sum_fp_fn = np.sum(mat[:,2]) + np.sum([mat[:,3]])
    micro_fscore = sum_tpos / (sum_tpos + 0.5*sum_fp_fn + epsilon)
    return micro_fscore, macro_fscore




# main for example
if __name__ == '__main__':
    pred = np.asarray([2 ,0 ,2 ,1 ,2 ,0 ,2 ,2, 1, 1])
    tst = np.asarray([2, 1, 0, 2, 2, 2, 0, 1, 1, 1])
    print(pred)
    print(tst)
    print(c_micro_macro_fscore(pred,tst,3))