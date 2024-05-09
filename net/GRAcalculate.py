# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler,MinMaxScaler
#
# #  某一列当做参照序列，其他为对比序列
# def graOne(Data,m=0):
#     """
#     return:
#     """
#     columns = Data.columns.tolist()  # 将列名提取出来
#     #第一步：无量纲化
#     data = MinMaxScaler().fit_transform(Data)
#     data = pd.DataFrame(data,columns=Data.columns)
#     referenceSeq  = data.iloc[:,m]  #参考序列
#     data.drop(columns[m],axis=1,inplace=True) # 删除参考列
#     compareSeq = data.iloc[:,0:]  #对比序列
#     row,col = compareSeq.shape
#     #第二步：参考序列 - 对比序列
#     data_sub = np.zeros([row,col])
#     for i in range(col):
#         data_sub[:,i] = abs(referenceSeq[:]-compareSeq.iloc[:,i])
#     #找出最大值和最小值
#     maxVal = np.max(data_sub)
#     minVal = np.min(data_sub)
#     cisi = np.zeros([row,col])
#     for i  in range(row):
#         cisi[i,:] = (minVal+0.5*maxVal) /(data_sub[i,:]+0.5*maxVal)
#     #第三步：计算关联度
#     result = [np.mean(cisi[:,i]) for i in range(col)]
#     result.insert(m,1) #参照列为1
#     return pd.DataFrame(result)
#
# def GRA(Data):
#     df = Data.copy()
#     columns = [str(s) for s in df.columns if s not in [None]] #[1 2 ,,,12]
#     # print(columns)
#     df_local = pd.DataFrame(columns=columns)
#     df.columns =columns
#     for i in range(len(df.columns)): #每一列都做参照序列，求关联系数
#         # print(i)
#         df_local.iloc[:,i] = graOne(df,m=i)[0]
#     df_local.index = columns
#     return df_local


# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os


class GraModel():
    '''灰色关联度分析模型'''

    def __init__(self, inputData, p=0.5, standard=True):
        '''
        初始化参数
        inputData：输入矩阵，纵轴为属性名，第一列为母序列
        p：分辨系数，范围0~1，一般取0.5，越小，关联系数间差异越大，区分能力越强
        standard：是否需要标准化
        '''
        self.inputData = inputData
        self.p = p
        self.standard = standard
        # 标准化
        self.standarOpt()
        # 建模
        self.buildModel()

    def standarOpt(self):
        '''标准化输入数据'''
        if not self.standard:
            return None
        self.scaler = MinMaxScaler().fit(np.array(self.inputData))
        self.scalerData = self.scaler.transform(np.array(self.inputData))

    def buildModel(self):
        # 第一列为母列，与其他列求绝对差
        momCol = self.scalerData[:, 0].copy()
        sonCol = self.scalerData[:, 0:].copy()
        for col in range(sonCol.shape[1]):
            sonCol[:, col] = abs(sonCol[:, col] - momCol)
        # 求两级最小差和最大差
        minMin = sonCol.min()
        maxMax = sonCol.max()
        # 计算关联系数矩阵
        cors = (minMin + self.p * maxMax) / (sonCol + self.p * maxMax)
        # 求平均综合关联度
        meanCors = pd.DataFrame(cors.mean(axis=0),index=self.inputData.columns)
        meanCors.columns = [self.inputData.columns[0]]
        # self.result = meanCors.sort_values(by=[0],ascending=False)[:20]
        self.result = meanCors

if __name__ == "__main__":
    from net.loadDataUtils import *
    from net.dealDataUtils import *
    basePath = r'D:\learningResource\researchResource\多目标优化\2021年D题\2021年D题'
    path = os.path.join(basePath, 'Molecular_Descriptor.xlsx')
    sheetName = ['training', 'test']
    pathY1 = os.path.join(basePath, 'ERα_activity.xlsx')
    pathY2 = os.path.join(basePath, 'ADMET.xlsx')
    ########################超参数设置#########################
    # 读取数据
    print("______________1.加载数据_________________")
    dataX = loadData(path, sheetName[0])
    dataY = loadData(pathY1, sheetName[0])
    dataY2 = loadData(pathY2, sheetName[0])
    # 1. ADMET数据标签处理，1代表对身体好，0代表对身体不好
    print("______________2.标签处理_________________")
    dataY2 = dealLableData(dataY2)
    # 2.根据相关性系数进行分子描述符筛选
    allIndex = []
    YColumns = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
    # ADMET 5个性质
    for i in range(5):
        print("-----" + YColumns[i] + "计算进行中" + str(i + 1) + "-----")
        Y = dataY2.iloc[:, i + 1]
        # 选取相关度最高的前10个自变量
        data = pd.concat([Y, dataX.iloc[:, 1:]], axis=1)
        # 灰色关联分析
        model = GraModel(data, standard=True)
        print(model.result)


