import pandas as pd
import numpy as np

def analysisData(dataX,dataY):
    # print("样本类别统计：",dataX.value_counts())
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

def dealDataX(data):
    # print(data.head(5))
    print(data.shape)
    # 1.删除90%以上都是0的特征
    row_count = data.shape[0]
    columns_to_drop = []
    for column, count in data.apply(lambda column: (column == 0).sum()).iteritems():
        if count / row_count >= 0.9:
            columns_to_drop.append(column)

    data.drop(columns_to_drop, axis=1, inplace=True)
    print(data.shape)
    # 2. 删除相关系数大于95%的特征
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    # print(to_drop)
    data.drop(to_drop, axis=1, inplace=True)
    print(data.shape)
    return data