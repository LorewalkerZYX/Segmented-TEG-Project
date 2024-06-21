import numpy as np


# Mq = 8.401629926494419
# Sq = 1.050063542837697

Mp = 1.409519676582611 
Sp = 1.949743019380285


def recover(y):
    # y /= const
    for i in range(len(y)):
        y[i] = y[i] * Sp + Mp
    outy = np.exp(y)
    return outy


def normalize(x, input=True):
    temp = x

    if input:
        hte = 9  # hte = [1-10]
        ff = 0.9  # ff = [0.05-0.95]
        n_ratio = 0.9  # n_ratio = [0.05-0.95]
        p_ratio = 0.9  # p_ratio = [0.05-0.95]
        rhoc_h = 9.9E-8  # rhoc_h = [1E-9-1E-7]
        rhoc_l = 9.9E-8  # rhoc_l = [1E-9-1E-7]
        qin = 380  # Qin = [20-400]*5

        for i in range(len(temp)):
            temp[i, 0] = (temp[i, 0] - 1) / hte
            temp[i, 1] = (temp[i, 1] - 0.05) / ff
            temp[i, 2] = (temp[i, 2] - 0.05) / n_ratio
            temp[i, 3] = (temp[i, 3] - 0.05) / p_ratio
            temp[i, 4] = (temp[i, 4] - 1E-9) / rhoc_h
            temp[i, 5] = (temp[i, 5] - 1E-9) / rhoc_l
            temp[i, 6] = (temp[i, 6]/5 - 20) / qin

    else:
        for k in range(len(temp)):
            temp[k] = np.log(temp[k])
            temp[k] = (temp[k] - Mp) / Sp
    return temp
