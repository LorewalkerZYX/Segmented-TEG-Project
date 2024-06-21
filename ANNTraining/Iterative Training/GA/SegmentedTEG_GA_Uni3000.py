# Segmented TEG Constant Heat Flux experiment
# GA Uni3000 Generate new 1000 dataset
# Available on https://github.com/LorewalkerZYX/Segmented-TEG-Project.git

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sko.GA import GA
import xlsxwriter
import time
import random

Batch_size = 64
epoch = 2000
learning_rate = 0.001
hidden_feature = 200

Mp = 1.409519676582611 
Sp = 1.949743019380285


# Set the random seed manually for reproducibility.
def seed_torch(seed=1029):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


seed_torch(2)


def recover_y(y):
    y[0] = y[0] * Sp + Mp
    outy = np.exp(y)
    return outy[0]


def selection_tournament(self, tourn_size=4):
    '''
    Select the best individual among *tournsize* randomly chosen
    individuals,
    :param self:
    :param tourn_size:
    :return:
    '''
    FitV = self.FitV
    sel_index = []
    for i in range(self.size_pop):
        aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
        # aspirants_index = np.random.randint(self.size_pop, size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom


def ranking(self):
    # GA select the biggest one, but we want to minimize func, so we put a negative here
    self.FitV = (self.Y - np.argmin(self.Y))  # self.Y  # [np.argsort(1 - self.Y)]
    return self.FitV


# create net
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layer):
        super(Net, self).__init__()
        self.input = nn.Linear(n_feature, n_hidden)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(n_hidden, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(n_hidden, n_output)
        self.layernum = n_layer

    def forward(self, x):
        out = self.input(x)
        out = self.relu(out)
        for i in range(self.layernum):
            out = self.hidden(out)
            out = self.relu(out)
        out = self.out(out)
        return out


TEG_NET = Net(7, hidden_feature, 1, 5)
TEG_NET.load_state_dict(torch.load('TEGNetSegQR0\'.pkl'))

'''
# normalization
def normalize_x(x, input=True):
    temp = x

    if input:
        wn = 4.5  # wn = [0.5-5]
        wp = 4.5  # wp = [0.5-5]
        h = 4.5  # h = [0.5-5]
        h_ic = 2.5  # h_ic = [0.5-3]
        ff = 0.9  # ff = [0.05-0.95]
        t_h = 200  # T_H = [300-500]
        rho_c = 9.9E-8  # rho_c = [1E-9-1E-7]

        for i in range(len(temp)):
            temp[i, 0] = (temp[i, 0] - 0.5) / wn
            temp[i, 1] = (temp[i, 1] - 0.5) / wp
            temp[i, 2] = (temp[i, 2] - 0.5) / h
            temp[i, 3] = (temp[i, 3] - 0.5) / h_ic
            temp[i, 4] = (temp[i, 4] - 0.05) / ff
            temp[i, 5] = (temp[i, 5] - 300) / t_h
            temp[i, 6] = (temp[i, 6] - 1E-9) / rho_c

    else:
        for k in range(len(temp)):
            temp[k][0] = np.log(temp[k][0])
            temp[k][0] = (temp[k][0] - MeR1) / SeR1
            temp[k][1] = np.log(temp[k][1])
            temp[k][1] = (temp[k][1] - MpR1) / SpR1
    return temp
'''


def normalize_new(x, input=True):
    temp = x
    hte = 9  # hte = [1-10]
    ff = 0.9  # ff = [0.05-0.95]
    n_ratio = 0.9  # n_ratio = [0.05-0.95]
    p_ratio = 0.9  # p_ratio = [0.05-0.95]
    rhoc_h = 9.9E-8  # rhoc_h = [1E-9-1E-7]
    rhoc_l = 9.9E-8  # rhoc_l = [1E-9-1E-7]
    qin = 1900  # Qin = [100-2000]
    temp[0] = (temp[0] - 1) / hte
    temp[1] = (temp[1] - 0.05) / ff
    temp[2] = (temp[2] - 0.05) / n_ratio
    temp[3] = (temp[3] - 0.05) / p_ratio
    temp[4] = (temp[4] - 1E-9) / rhoc_h
    temp[5] = (temp[5] - 1E-9) / rhoc_l
    temp[6] = (temp[6] - 100) / qin
    return temp


Th = 0.5  # Th = 400
R_c = 1/11  # Rho_c = 1E-8
'''
T_H = 435
Rhoc_h = 8.9E-8
Rhoc_l = 3.5E-8
'''

def demo_func(x):
    # print(x[0, :])
    # x.reshape(4, 300)
    # temp = normalize_x(x)
    y = x / 100
    # y[5] *= 100
    # y[4] /= 10
    y = np.append(y, Rhoc_h)
    y = np.append(y, Rhoc_l)
    y = np.append(y, Qin)
    InputX = normalize_new(y)
    temp = torch.Tensor(InputX)
    # x1, x2, x3, x4 = temp
    # InX = normalize_new(temp)
    # print(InX)
    result = TEG_NET(temp)
    tempy = result.cpu().data.numpy()
    outy = recover_y(tempy)
    return outy


def maxrun(self, max_iter=None):
    self.max_iter = max_iter or self.max_iter
    for i in range(self.max_iter):
        self.X = self.chrom2x(self.Chrom)
        self.Y = self.x2y()
        self.ranking()
        self.selection()
        self.crossover()
        self.mutation()

        # record the best ones
        generation_best_index = self.FitV.argmax()
        self.generation_best_X.append(self.X[generation_best_index, :])
        self.generation_best_Y.append(self.Y[generation_best_index])
        self.all_history_Y.append(self.Y)
        self.all_history_FitV.append(self.FitV)

    global_best_index = np.array(self.generation_best_Y).argmax()
    self.best_x = self.generation_best_X[global_best_index]
    self.best_y = self.func(np.array([self.best_x]))
    return self.best_x, self.best_y


leastB = [0, 0, 0, 0, 0]
MostB = [1, 1, 1, 1, 1]

leastB1 = [100, 5, 5, 5]
MostB1 = [1000, 95, 95, 95]
# hte, ff, n_ratio, p_ratio
#


GA.run = maxrun


workbook = xlsxwriter.Workbook('GA_Qin_R0_new.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1', 'HTE')
worksheet.write('B1', 'FF')
worksheet.write('C1', 'N_Ratio')
worksheet.write('D1', 'P_Ratio')
worksheet.write('E1', 'Rhoc_h')
worksheet.write('F1', 'Rhoc_l')
worksheet.write('G1', 'Qin')
worksheet.write('H1', 'PowerDensity')


NData = pd.read_excel('Qin_GA_1.xlsx')
OldD = NData.iloc[:, :]
New = OldD.to_numpy()


for i in range(len(New)):
    ga = GA(
        func=demo_func,
        n_dim=4, size_pop=100,
        max_iter=200,
        lb=leastB1,
        ub=MostB1,
        precision=1,
        prob_mut=0.01
    )
# ga.register(operator_name='selection', operator=selection_tournament)
    ga.register(operator_name='ranking', operator=ranking)
    Rhoc_h = New[i][0]
    Rhoc_l = New[i][1]
    Qin = New[i][2]
    best_x, best_y = ga.run()
# write in Excel
    worksheet.write(i+1, 0, best_x[0]/100)
    worksheet.write(i+1, 1, best_x[1]/100)
    worksheet.write(i+1, 2, best_x[2]/100)
    worksheet.write(i+1, 3, best_x[3]/100)
    worksheet.write(i+1, 4, Rhoc_h)
    worksheet.write(i+1, 5, Rhoc_l)
    worksheet.write(i+1, 6, Qin)
    worksheet.write(i+1, 7, best_y[0])
    id = (i+1)/10
    print('\r', ' Finishing %.2f %%' % id)
workbook.close()
