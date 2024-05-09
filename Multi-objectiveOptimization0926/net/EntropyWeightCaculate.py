import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pygad.torchga
import pygad
import os
import pandas as pd
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier

def loadData(path,sheetName):
    data = pd.read_excel(path,sheetName)
    return data

def analysisData(dataX,dataY):
    print(dataX.head(5))
    print(dataX.shape)
    print(dataY.head(5))
    print(dataY.shape)

def dealData(dataX,dataY):
    dataX = dataX.iloc[:,1:]
    print(dataX.head(2))
    dataY = dataY.iloc[:,-1]
    print(dataY.head(2))
    data = pd.concat([dataY,dataX],axis=1)
    print(data.head(2))
    print(data.shape)
    return data


def dealLableData(dataY2):
    dataY2['hERG'].replace(0,-1,inplace=True)
    dataY2['hERG'].replace(1, 0,inplace=True)
    dataY2['hERG'].replace(-1, 1,inplace=True)
    dataY2['MN'].replace(0, -1, inplace=True)
    dataY2['MN'].replace(1, 0, inplace=True)
    dataY2['MN'].replace(-1, 1, inplace=True)
    return dataY2

def entropy(data0):
    #返回每个样本的指数
    #样本数，指标个数
    n,m=np.shape(data0)
    #一行一个样本，一列一个指标
    #下面是归一化
    maxium=np.max(data0,axis=0)
    minium=np.min(data0,axis=0)
    data= (data0-minium)*1.0/(maxium-minium)
    ##计算第j项指标，第i个样本占该指标的比重
    sumzb=np.sum(data,axis=0)
    data=data/sumzb
    #对ln0处理
    a=data*1.0
    a[np.where(data==0)]=0.0001
#    #计算每个指标的熵
    e=(-1.0/np.log(n))*np.sum(data*np.log(a),axis=0)
#    #计算权重
    w=(1-e)/np.sum(1-e)
    recodes=np.sum(data0*w,axis=1)
    return recodes

if __name__ == "__main__":
    device = torch.device( 'cpu')
    basePath = r'D:\learningResource\researchResource\多目标优化\2021年D题\2021年D题'
    path = os.path.join(basePath, 'Molecular_Descriptor.xlsx')
    sheetName = ['training', 'test']
    pathY1 = os.path.join(basePath, 'ERα_activity.xlsx')
    pathY2 = os.path.join(basePath, 'ADMET.xlsx')
    #读取数据
    dataX = loadData(path,sheetName[0])
    dataY = loadData(pathY1,sheetName[0])
    dataY2 = loadData(pathY2, sheetName[0])
    #1. ADMET数据标签处理，1代表对身体好，0代表对身体不好
    dataY2 = dealLableData(dataY2)
    #2.计算各个函数f(x)及g(x)的权值
    data = pd.concat([dataY.iloc[:,2],dataY2.iloc[:,1:]],axis=1)
    data = MinMaxScaler().fit_transform(data)
    entroD = entropy(np.array(data).T)
    print(entroD)