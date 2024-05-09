import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN,BorderlineSMOTE
from imblearn.combine import SMOTEENN,SMOTETomek
from collections import Counter
from sklearn.utils import shuffle

def splitData00(data,isPCA=False,isSample=False):
    n_components = 25
    features,lables = data.iloc[:,1:],data.iloc[:,0]
    #过采样
    if isSample:
        ros = SMOTEENN(random_state=0)#RandomOverSampler(random_state=0)
        print("样本类别统计：", lables.value_counts())
        features,lables = ros.fit_resample(features,lables)
        print("过采样结果：",sorted(Counter(lables).items()))
    #数据标准化
    features = StandardScaler().fit_transform(features)
    #特征压缩（降维），降低BP神经网络过拟合的风险
    if isPCA:
        pca = PCA(n_components=n_components)
        # features = pca.fit_transform(features)
        # print("使用PCA降至维度：{},降维后信息保留度：{}".format(n_components,sum(pca.explained_variance_ratio_)))
    features = pd.DataFrame(features)
    XTrain, XTest, YTrain, YTest = train_test_split(features, lables, test_size=0.2,shuffle=True, random_state=32)
    # XTrain, XTest, YTrain, YTest = features, features, lables, lables
    return XTrain, XTest, YTrain, YTest,n_components,pca

def ORsplitData(data,isPCA=False):
    n_components = 25
    # data = shuffle(data)
    features,lables = data.iloc[:,1:].to_numpy(),data.iloc[:,0].to_numpy()
    # 数据标准化
    # features = StandardScaler().fit_transform(features)
    # 特征压缩（降维），降低BP神经网络过拟合的风险
    if isPCA:
        pca = PCA(n_components=n_components)
        # features = pca.fit_transform(features)
        # print("使用PCA降至维度：{},降维后信息保留度：{}".format(n_components,sum(pca.explained_variance_ratio_)))
    # features = pd.DataFrame(features)
    XTrain, XTest, YTrain, YTest = train_test_split(features,lables,test_size=0.2,shuffle=True,random_state=32)
    ss = StandardScaler()
    XTrain = ss.fit_transform(XTrain)
    XTest = ss.transform(XTest)
    return XTrain, XTest, YTrain, YTest, n_components, pca

def splitData(data,type='c'):
    data = shuffle(data)
    features,lables = data.iloc[:,1:].to_numpy(),data.iloc[:,0].to_numpy()
    # if type == 'c':
    #     ros = SMOTEENN(random_state=0)  # RandomOverSampler(random_state=0)
    #     # print("样本类别统计：", pd.DataFrame(lables).value_counts())
    #     features, lables = ros.fit_resample(features, lables)
    #     # print("样本类别统计：", pd.DataFrame(lables).value_counts())
    return features,lables

def KFlodSplitData(XX, YY,train_ids, test_ids,type='c'):
    X_train = XX[train_ids]
    Y_train = YY[train_ids]
    X_test = XX[test_ids]
    Y_test = YY[test_ids]
    # if type == 'c':
    #     ros = BorderlineSMOTE(random_state=42, kind='borderline-2')
    #     #     ros = SMOTE(random_state=0)  # RandomOverSampler(random_state=0)
    #     X_train, Y_train = ros.fit_resample(X_train, Y_train)
    scaleX = StandardScaler()
    X_train = scaleX.fit_transform(X_train)
    X_test = scaleX.transform(X_test)

    # scaleY = StandardScaler()
    # scaleY.fit(Y_train.reshape(-1,1))
    # Y_train = scaleY.fit_transform(Y_train.reshape(-1,1))
    return X_train, Y_train,X_test,Y_test

def splitDataXR(X,Y):
    XTrain, XTest, YTrain, YTest = train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=32)
    ss = StandardScaler()
    XTrain = ss.fit_transform(XTrain)
    XTest = ss.transform(XTest)
    return XTrain, XTest, YTrain, YTest

