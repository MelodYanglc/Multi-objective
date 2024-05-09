import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering,KMeans
from sklearn import metrics
# from communities.algorithms import louvain_method,girvan_newman,spectral_clustering
from net.GRAcalculate import *
import matplotlib.pyplot as plt
import torch.optim as optim
from net.GRAcalculate import *
from net.splitDataUtils import *
from net.BPmodel import *
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold,KFold,RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
import pickle #pickle模块


def loadData(path,sheetName):
    data = pd.read_excel(path,sheetName)
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

def caculate_corr(data):
    res = data.corr()
    res[np.isnan(res)] = 0
    return res

def caculate_kendall_corr(data):
    res = data.corr('kendall')
    res[np.isnan(res)] = 0
    return res

def caculate_spearman_corr(data):
    res = data.corr('spearman')
    res[np.isnan(res)] = 0
    return res

def caculate_kl_divergence(da):
    data = da.T
    res = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            res[i][j] = scipy.stats.entropy(data[:][i], data[:][j])
    res[np.isneginf(res)] = 0
    res[np.isnan(res)] = 0
    # scale = MinMaxScaler()
    # scale.fit_transform(res)
    return res

def caculate_l2_distance(data):
    data = StandardScaler().fit_transform(data)
    data = data.T
    res = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            res[i][j] = np.linalg.norm(data[:][i] - data[:][j])
    res[np.isneginf(res)] = 0
    res[np.isnan(res)] = 0
    return res

def caculate_cosineSimilarity(data):
    data = MinMaxScaler().fit_transform(data)
    num = np.dot(np.array(data).T,data)  # 向量点乘
    denom = np.linalg.norm(np.array(data).T, axis=1).reshape(-1, 1) * np.linalg.norm(np.array(data).T, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    res[np.isnan(res)] = 0
    return res

def caculate_Gra(data,p=0.5):
    # print(da.shape)
    result = np.zeros((data.shape[1], data.shape[1]))
    data = StandardScaler().fit_transform(data)
    for i in range(data.shape[1]):
        momCol = data[:,i].copy()
        res = np.zeros((data.shape[0], data.shape[1]))
        for j in range(data.shape[1]):
            res[:, j] = abs(data[:, j] - momCol)

        # 求两级最小差和最大差
        minMin = res.min()
        maxMax = res.max()
        # 计算关联系数矩阵
        cors = (minMin + p * maxMax) / (res + p * maxMax)
        # 求平均综合关联度
        meanCors = cors.mean(axis=0)
        result[:,i] = meanCors
    return result


def p_Means(data):
    res = SpectralClustering().fit_predict(np.array(data))
    return res

def caculate_maxG(data,RelativeMatrix,clusters):
    selected_G =[]
    clusters_center = [i for i in range(len(set(clusters)))]
    mask = np.zeros((RelativeMatrix.shape[1],RelativeMatrix.shape[1]))
    for c in clusters_center:
        res_list = [idx for idx,x in list(enumerate(clusters)) if x == c]
        for k in res_list:
            mask[k][:] = 1

        select_matirx = np.multiply(RelativeMatrix,mask)
        # print("data:",data[:5,:5])
        # print("mask:",mask[:5, :5])
        # print("select_matirx:",select_matirx[:5,:5])
        each_max_node_D = -1024
        each_max_node_index = -1
        for k in res_list:
            # print("std:",np.std(data.iloc[:,k]))
            # count_node_D = np.std(data.iloc[:,k]) / sum(select_matirx[k][:])
            # count_node_D = np.std(data.iloc[:,k]) * sum(select_matirx[k][:])
            count_node_D =  sum(select_matirx[k][:])
            if count_node_D > each_max_node_D:
                each_max_node_D = count_node_D
                each_max_node_index = k
        selected_G.append(each_max_node_index)
    return selected_G

def caculateFeatureImportance(data):
    stdF = data.std()
    stdF = np.array(stdF).reshape(1,len(stdF))
    stdFF = np.repeat(stdF, repeats=stdF.shape[1], axis=0)

def turn_arg(X, k):
    # 寻找最合适的参数gamma

    # 默认使用的是高斯核，需要对n_cluster和gamma进行调参，选择合适的参数
    scores = []
    s = dict()
    for index, gamma in enumerate((0.001,0.01, 0.1, 1, 10)):
        pred_y = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
        print("Calinski-Harabasz Score with k=" + str(k) + " gamma=", gamma, "score=",
              metrics.calinski_harabasz_score(X, pred_y))
        tmp = dict()
        tmp['gamma'] = gamma
        tmp['score'] = metrics.calinski_harabasz_score(X, pred_y)
        s[metrics.calinski_harabasz_score(X, pred_y)] = tmp
        scores.append(metrics.calinski_harabasz_score(X, pred_y))
    print(np.max(scores))
    print("最大得分项：")
    print(s.get(np.max(scores)))
    gamma = s.get(np.max(scores))['gamma']
    return gamma

def modelParametersSelect(rf,XX, YY):
    from sklearn.metrics import mean_squared_error, accuracy_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    mseTrain= []
    mseTest = []
    for i in range(5,80):
        rf = RandomForestClassifier(max_leaf_nodes=i)
        for train_id, test_id in sfolder.split(XX, YY):
            X_train, Y_train, X_test, Y_test = KFlodSplitData(XX, YY, train_id, test_id)
            rf.fit(X_train, Y_train)
            predTrain = rf.predict(X_train)
            predTest = rf.predict(X_test)
            trainA = accuracy_score(predTrain,Y_train)
            testA = accuracy_score(predTest, Y_test)
            mseTrain.append(trainA)
            mseTest.append(testA)

    x = range(5,80)
    plt.plot(x,mseTrain,'*-')
    plt.plot(x,mseTest,'*-')
    plt.show()

def describeData(data):
    print(type(data))
    # data = pd.DataFrame(data)
    print(data.describe())
    print(data.info())
    for i in range(1,data.shape[1]):
        data.iloc[:, i].plot.box(title="Box Chart")
        plt.grid(linestyle="--", alpha=0.3)
        plt.show()

def three_sigma(Ser1):
    '''
    Ser1：表示传入DataFrame的某一列。
    '''
    rule = (Ser1.mean()-3*Ser1.std()>Ser1) | (Ser1.mean()+3*Ser1.std()< Ser1)
    index = np.arange(Ser1.shape[0])[rule]
    return index  #返回落在3sigma之外的行索引值

def delete_out3sigma(data):

    out_index = [] #保存要删除的行索引
    for i in range(data.shape[1]): # 对每一列分别用3sigma原则处理
        index = three_sigma(data.iloc[:,i])
        out_index += index.tolist()
    delete_ = list(set(out_index))
    print(len(delete_))
    print('所删除的行索引为：',delete_)
    data.drop(delete_,inplace=True)
    return data

def CountErrorFeatureNum(data):
    columns = data.columns
    # print(columns)
    for c in columns[1:-1]:  # 对每一列分别用3sigma原则处理
        # print(data)
        eachMean = data[c].mean()
        eachStd = data[c].std()
        # print(c,eachMean,eachStd)
        maxR = eachMean + 3 * eachStd
        minR = eachMean - 3 * eachStd
        data = data[data[c] >= minR]
        data = data[data[c] <= maxR]
    return data

def modelHGB(type='c',cv=5,shuffle=True):
    from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
    if type == 'c':
        rf = HistGradientBoostingClassifier()
        sfolder = StratifiedKFold(n_splits=cv, random_state=0, shuffle=shuffle)
    else:
        rf = HistGradientBoostingRegressor()
        sfolder = KFold(n_splits=cv, random_state=0, shuffle=shuffle)
    # scores = cross_val_score(rf, XTrain, YTrain, cv=cv)

    return rf,sfolder

def modelLGBM(type='c',cv=5,shuffle=True):
    from sklearn.linear_model import LinearRegression,LogisticRegression
    from sklearn.metrics import mean_squared_error,accuracy_score,f1_score
    if type == 'c':
        rf = LGBMClassifier()
        sfolder = StratifiedKFold(n_splits=cv, random_state=0, shuffle=shuffle)
    else:
        rf = LGBMRegressor()
        sfolder = KFold(n_splits=cv, random_state=0, shuffle=shuffle)
    # scores = cross_val_score(rf, XTrain, YTrain, cv=cv)
    return rf,sfolder

def modelRF(type='c',cv=5,shuffle=True):
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
    if type == 'c':
        rf = RandomForestClassifier(
            # max_leaf_nodes=30
        )
        sfolder = StratifiedKFold(n_splits=cv, random_state=0, shuffle=shuffle)
    else:
        rf = RandomForestRegressor(
            # max_leaf_nodes=30
        )
        sfolder = KFold(n_splits=cv, random_state=0, shuffle=shuffle)
    # scores = cross_val_score(rf, XTrain, YTrain, cv=cv)

    return rf,sfolder

def modelADA(type='c',cv=5,shuffle=True):
    from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
    from sklearn.metrics import mean_squared_error,accuracy_score
    if type == 'c':
        rf = AdaBoostClassifier()
        sfolder = StratifiedKFold(n_splits=cv, random_state=0, shuffle=shuffle)
    else:
        rf = AdaBoostRegressor()
        sfolder = KFold(n_splits=cv, random_state=0, shuffle=shuffle)

    # scores = cross_val_score(rf, XTrain, YTrain, cv=cv)
    return rf,sfolder

def modelGBDT(type='c',cv=5,shuffle=True):
    from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error,accuracy_score
    if type == 'c':
        rf = GradientBoostingClassifier()
        sfolder = StratifiedKFold(n_splits=cv, random_state=0, shuffle=shuffle)
    else:
        rf = GradientBoostingRegressor()
        sfolder = KFold(n_splits=cv, random_state=0, shuffle=shuffle)
    # scores = cross_val_score(rf, XTrain,YTrain,cv=cv)
    return rf,sfolder

def modelxgb(type='c',cv=5,shuffle=True):
    from xgboost import XGBRegressor,XGBRFClassifier
    from sklearn.metrics import mean_squared_error,accuracy_score
    if type == 'c':
        rf = XGBRFClassifier()
        sfolder = StratifiedKFold(n_splits=cv, random_state=0, shuffle=shuffle)
    else:
        rf = XGBRegressor()
        sfolder = KFold(n_splits=cv, random_state=0, shuffle=shuffle)
    # scores = cross_val_score(rf, XTrain, YTrain, cv=cv)
    return rf,sfolder

def modelSVM(type='c',cv=5,shuffle=True):
    from sklearn.svm import SVR,SVC
    from sklearn.metrics import mean_squared_error,accuracy_score
    if type == 'c':
        rf = SVC()
        sfolder = StratifiedKFold(n_splits=cv, random_state=0, shuffle=shuffle)
    else:
        rf = SVR()
        sfolder = KFold(n_splits=cv, random_state=0, shuffle=shuffle)
    # scores = cross_val_score(rf, XTrain, YTrain, cv=cv)
    return rf,sfolder

def modelLR(type='c',cv=5,shuffle=True):
    from sklearn.linear_model import LinearRegression,LogisticRegression
    from sklearn.metrics import mean_squared_error,accuracy_score,f1_score
    if type == 'c':
        rf = LogisticRegression()
        sfolder = StratifiedKFold(n_splits=cv, random_state=0, shuffle=shuffle)
    else:
        rf = LinearRegression()
        sfolder = KFold(n_splits=cv, random_state=0, shuffle=shuffle)
    # scores = cross_val_score(rf, XTrain, YTrain, cv=cv)
    return rf,sfolder

def modeMLP(type='c',cv=5,shuffle=True):
    from sklearn.neural_network import MLPRegressor,MLPClassifier
    from sklearn.metrics import mean_squared_error,accuracy_score,f1_score
    if type == 'c':
        rf = MLPClassifier(
            hidden_layer_sizes=(100,60),learning_rate_init=0.001,learning_rate='adaptive',solver='adam',momentum=0.9,alpha=2
        )
        sfolder = StratifiedKFold(n_splits=cv, random_state=0, shuffle=shuffle)
    else:
        rf = MLPRegressor(
            hidden_layer_sizes=(100,60),learning_rate_init=0.001,learning_rate='adaptive',solver='adam',momentum=0.9,alpha=9,
        )
        sfolder = KFold(n_splits=cv, random_state=0, shuffle=shuffle)
    # scores = cross_val_score(rf, XTrain, YTrain, cv=cv)
    return rf,sfolder

def trainAndEvalModel(rf,sfolder,X, Y,type = 'c'):
    from sklearn.metrics import mean_squared_error,accuracy_score,f1_score,recall_score
    measureResultTrain = []
    measureResultTest = []
    for train_id, test_id in sfolder.split(X, Y):
        X_train, Y_train, X_test, Y_test = KFlodSplitData(X, Y, train_id, test_id,type=type)
        nter(Y_train).items()))

        rf.fit(X_train,Y_train)
        pre_YTrain = rf.predict(X_train)
        pre_YTest = rf.predict(X_test)
        # print(rf.get_params())
        if type == 'c':
            measureResultTrain.append(accuracy_score(pre_YTrain, Y_train))
            measureResultTest.append(accuracy_score(pre_YTest, Y_test))
            # print(classification_report(pre_YTrain, Y_train))
        else:
            measureResultTrain.append(mean_squared_error(pre_YTrain, Y_train))
            measureResultTest.append( mean_squared_error(pre_YTest, Y_test) )
    print(np.mean(measureResultTrain), np.mean(measureResultTest))
    return np.mean(measureResultTrain), np.mean(measureResultTest)

def trainAndEvalModel2(rf,sfolder,X, Y,type = 'c'):
    measureResultTrain = []
    measureResultTest = []
    if type == 'c':
        cv = StratifiedKFold(n_splits=10,  random_state=1)
        n_scores = cross_val_score(rf, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
        measureResultTrain.append(n_scores)
        measureResultTest.append(n_scores)
    return np.mean(measureResultTrain), np.mean(measureResultTest)

def trainAndEvalModel3(rf,sfolder,X, Y,type = 'c'):
    from sklearn.metrics import mean_squared_error,accuracy_score,f1_score,recall_score
    measureResultTrain = []
    measureResultTest = []

    if type == 'c':
        from imblearn.over_sampling import SMOTE, ADASYN
        ros =  SMOTE()
        XTrain,YTrain = ros.fit_resample(XTrain,YTrain)

    rf.fit(XTrain,YTrain)

    predTrain = rf.predict(XTrain)
    predTest = rf.predict(XTest)
    if type == 'c':
        measureResultTrain.append(accuracy_score(predTrain, YTrain))
        measureResultTest.append(accuracy_score(predTest, YTest))
    else:
        measureResultTrain.append(mean_squared_error(predTrain, YTrain))
        measureResultTest.append(mean_squared_error(predTest, YTest))
    return np.mean(measureResultTrain), np.mean(measureResultTest)


if __name__=="__main__":
    #######################超参数设置########################
    device = "cuda"  # torch.device('cpu')
    clustersNumber = 20
    epochs = 200
    learningRate1 = 0.001
    learningRate2 = 0.001
    batch_size1 = 256
    batch_size2 = 256
    step_size = 5
    flodNum = 5
    basePath = r'D:\learning resource\research&Life\多目标优化\dataSet'
    path = os.path.join(basePath, 'Molecular_Descriptor.xlsx')
    sheetName = ['training', 'test']
    pathY1 = os.path.join(basePath, 'ERα_activity.xlsx')
    pathY2 = os.path.join(basePath, 'ADMET.xlsx')
    pathDescr3 = os.path.join(basePath, 'DescriptorAll.xlsx')
    ########################超参数设置#########################
    print("______________1.加载数据_________________")
    dataX = loadData(path, sheetName[0])
    dataY = loadData(pathY1, sheetName[0])
    dataY2 = loadData(pathY2, sheetName[0])
    dataDescr3 = loadData(pathDescr3, 'Summary')
    print("______________1.数据描述_________________")
    # dataFF = pd.concat([dataX.iloc[:, 1:],dataY.iloc[:, -1]], axis=1)
    # describeData(dataFF)
    print("______________2.标签处理_________________")
    dataY2 = dealLableData(dataY2)
    print("______________2.异常值处理_________________")
    # dataX = CountErrorFeatureNum(dataX)
    # print(dataX.shape)
    dataX = dealDataX(dataX)
    # delete_out3sigma(dataY.iloc[:, -1])
    print("______________3.特征选择_________________")
    allIndex = []
    YColumns = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
    name_path = './modelTest0910nameAll/cluster' + str(clustersNumber *3) + '/names.txt'
    if not os.path.exists(name_path):
        data = dataX.iloc[:, 1:]
        # 相关性系数
        result1 = caculate_corr(data)
        # 余弦相似度
        result2 = caculate_cosineSimilarity(data)
        result3 = caculate_Gra(data)
        result = [result1,result2,result3]
        finnal_index = []
        n_clusters_select = [clustersNumber, clustersNumber, clustersNumber]
        # n_clusters_select = [2, 3, clustersNumber]
        k = 0
        for r in result:
            r = np.array(r)
            n = r.shape[0]
            r[range(n), range(n)] = 0
            model = SpectralClustering(n_clusters=n_clusters_select[k], affinity='rbf')
            k += 1
            communities = model.fit_predict(r)
            # plt.scatter(np.array(r)[:, 0], np.array(r)[:, 1], c=communities)
            # plt.show()
            # print(communities)
            selected_cluster_centers = caculate_maxG(data,r, communities)
            finnal_index.extend(selected_cluster_centers)
        finnal_index_all = list(set(finnal_index))
        print(finnal_index_all)
        print(len(finnal_index_all))
        columns = list(data.columns)
        finnal_index_all_name = [x for idx, x in enumerate(columns) if idx in finnal_index_all]
        print(finnal_index_all_name)
        # 保存文件
        with open(name_path, 'w') as f:
            f.write(str(finnal_index_all_name))
    print("______________4.特征独立性检验________________")
    # 读取文件
    with open(name_path, 'r') as f:
        content = f.readlines()[0].strip("[").strip("]").strip().split(",")
    # print("content:",content)
    names = [c.strip().strip("\'") for c in content]
    # print("names:", names)
    # print(type(names))
    data = dataX.iloc[:, 1:]
    finalX = data[names]
    # print(finalX)
    featuresNum = len(names)
    print("特征总数：", len(names))
    desTypeCount = []
    for index, des in enumerate(dataDescr3.iloc[:, 2]):
        for name1 in names:
            if name1 in des:
                desTypeCount.append(dataDescr3.iloc[index, 0])
    print("特征类别统计：", Counter(desTypeCount))
    print("特征类别数量：", len(set(desTypeCount)))

    print("______________4.模型训构建________________")
    # 读取文件
    with open(name_path, 'r') as f:
        content = f.readlines()[0].strip("[").strip("]").strip().split(",")
    # print("content:",content)
    names = [c.strip().strip("\'") for c in content]
    # print("names:", names)
    # print(type(names))
    data = dataX.iloc[:, 1:]
    # finalX = data[finnalNames]
    finalX = data[names]
    # print(finalX)
    featuresNum = len(names)
    print("特征总数：", len(names))

    AccTrain = []
    AccTest = []
    fucAll = []
    train_acc_allEpochs = []
    test_acc_allEpochs = []
    # 数据集切分
    data = pd.concat([dataY.iloc[:, -1], finalX], axis=1)
    data = CountErrorFeatureNum(pd.DataFrame(data))
    print(data.shape)
    XX, YY = splitData(data,type='r')
    print("________模型训练（PIC50回归）___________")
    # 模型训练
    rf,sfolder = modeMLP(type='r',cv=flodNum,shuffle=True)
    trainACC, testACC = trainAndEvalModel(rf,sfolder,XX, YY,type = 'r')
    AccTrain.append(trainACC)
    AccTest.append(testACC)
    # 3.2 ADMET 5个特征训练
    labelsCount = []
    for i in range(5):
        print("特征" + str(i + 1) + "计算过程")
        print("________数据集划分___________")
        Y = dataY2.iloc[:, i + 1]
        data = pd.concat([Y, finalX], axis=1)
        data = CountErrorFeatureNum(pd.DataFrame(data))
        print(data.shape)
        # 正负样本统计
        labelsCount.append(list(Y.value_counts()))
        # 3.1 数据集切分
        XX, YY = splitData(data,type='c')
        # 模型训练
        print("________模型训练（ADMET分类）___________")
        rf, sfolder = modeMLP(type='c', cv=flodNum,shuffle=True)

        # modelParametersSelect(rf, XX, YY)
        trainACC, testACC = trainAndEvalModel(rf, sfolder, XX, YY,type = 'c')
        # 存储数据
        AccTrain.append(trainACC)
        AccTest.append(testACC)

    print("train model MSE/ACC:{} avg acc:{} +/- {}".format(AccTrain,np.mean(AccTrain[1:]),np.std(AccTrain[1:])) )
    print("test model MSE/ACC:{} avg acc:{} +/- {}".format(AccTest,np.mean(AccTest[1:]),np.std(AccTest[1:])))
