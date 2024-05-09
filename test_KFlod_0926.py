import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering,KMeans
from sklearn import metrics
from net.GRAcalculate import *
import matplotlib.pyplot as plt
import torch.optim as optim
from net.GRAcalculate import *
from net.splitDataUtils import *
from net.plotData import *
from net.BPmodel import *
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from net.MOPModel import *

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

def caculate_l2_distance(da):
    data = da.T
    res = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            res[i][j] = np.linalg.norm(data[:][i] - data[:][j])
    res[np.isneginf(res)] = 0
    res[np.isnan(res)] = 0
    scale = MinMaxScaler()
    output = scale.fit_transform(res)
    return output

def caculate_cosineSimilarity(data):
    num = np.dot(np.array(data).T,data)  # 向量点乘
    denom = np.linalg.norm(np.array(data).T, axis=1).reshape(-1, 1) * np.linalg.norm(np.array(data).T, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    res[np.isnan(res)] = 0
    scale = MinMaxScaler()
    output = scale.fit_transform(res)
    return output

def caculate_Gra(da,p=0.5):
    # print(da.shape)
    result = np.zeros((da.shape[1], da.shape[1]))
    data = MinMaxScaler().fit_transform(da)
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

class SmoothCrossEntropy(nn.Module):
    """
    loss = SmoothCrossEntropy()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    """
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


def CountErrorFeatureNum(data):
    columns = data.columns
    for c in columns[1:]:  # 对每一列分别用3sigma原则处理
        # print(data)
        eachMean = data[c].mean()
        eachStd = data[c].std()
        # print(c,eachMean,eachStd)
        maxR = eachMean + 3 * eachStd
        minR = eachMean - 3 * eachStd
        dataF = data[data[c] < minR]
        dataF2 = data[data[c] > maxR]
        num = pd.concat([dataF,dataF2],axis=1).shape[0]
        if num > 100:
            data.drop([c],axis=0, inplace=True)
        else:
            data[c] = data[c].clip(lower=minR)
            data[c] = data[c].clip(upper=maxR)
    return data

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

def trainAndEvalModel(rf,sfolder,X, Y,type = 'c'):
    from sklearn.metrics import mean_squared_error,accuracy_score,f1_score,recall_score
    measureResultTrain = []
    measureResultTest = []
    for train_id, test_id in sfolder.split(X, Y):
        X_train, Y_train, X_test, Y_test = KFlodSplitData(X, Y, train_id, test_id,type=type)
        rf.fit(X_train,Y_train)
        pre_YTrain = rf.predict(X_train)
        pre_YTest = rf.predict(X_test)
        # print(rf.get_params())
        if type == 'c':
            measureResultTrain.append(accuracy_score(pre_YTrain, Y_train))
            measureResultTest.append(accuracy_score(pre_YTest, Y_test))
        else:
            measureResultTrain.append(mean_squared_error(pre_YTrain, Y_train))
            measureResultTest.append( mean_squared_error(pre_YTest, Y_test) )
    print(np.mean(measureResultTrain), np.mean(measureResultTest))
    return np.mean(measureResultTrain), np.mean(measureResultTest)

if __name__=="__main__":
    #######################超参数设置########################
    device = "cuda"#torch.device('cpu')
    clustersNumber = 20
    epochs = 150
    learningRate1 = 0.001
    learningRate2 = 0.001
    batch_size1 = 512
    batch_size2 = 512
    step_size = 5
    k_folds = 5
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
    print("______________2.标签处理_________________")
    dataY2 = dealLableData(dataY2)
    print("______________2.异常值处理_________________")
    # dataX = CountErrorFeatureNum(dataX)
    dataX = dealDataX(dataX)
    print("______________3.特征选择_________________")
    allIndex = []
    YColumns = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
    name_path = './nameAll/cluster'+str(clustersNumber*3)+'/names.txt'
    if not os.path.exists(name_path):
        data = dataX.iloc[:, 1:]
        # 相关性系数
        result1 = caculate_corr(data)
        # 余弦相似度
        result2 = caculate_cosineSimilarity(data)
        # 灰色关联度
        result3 = caculate_Gra(data)
        result = [result1,result2,result3]
        finnal_index = []
        n_clusters_select = [clustersNumber,clustersNumber,clustersNumber]
        k = 0
        for r in result:
            r = np.array(r)
            n = r.shape[0]
            r[range(n), range(n)] = 0
            model = SpectralClustering(n_clusters=n_clusters_select[k],affinity='nearest_neighbors')
            k += 1
            communities = model.fit_predict(np.array(r))
            selected_cluster_centers = caculate_maxG(data,r, communities)#caculate_maxG(np.array(r), communities)
            finnal_index.extend(selected_cluster_centers)
        finnal_index_all = list(set(finnal_index))
        print(finnal_index_all)
        print(len(finnal_index_all))
        columns = list(data.columns)
        finnal_index_all_name = [x for idx,x in enumerate(columns) if idx in finnal_index_all]
        print(finnal_index_all_name)
        #保存文件
        with open(name_path,'w') as f:
            f.write(str(finnal_index_all_name))
    print("______________4.特征独立性检验________________")
    #读取文件
    with open(name_path,'r') as f:
        content = f.readlines()[0].strip("[").strip("]").strip().split(",")
    # print("content:",content)
    names = [c.strip().strip("\'") for c in content]
    # print("names:", names)
    # print(type(names))
    data = dataX.iloc[:, 1:]
    finalX = data[names]
    # print(finalX)
    featuresNum = len(names)
    print("特征总数：",len(names))
    print(finalX.shape)
    # 依拉达准则清除异常数据
    finalX = CountErrorFeatureNum(finalX)
    print(finalX.shape)
    desTypeCount = []
    for index, des in enumerate(dataDescr3.iloc[:, 2]):
        for name1 in names:
            if name1 in des:
                desTypeCount.append(dataDescr3.iloc[index, 0])
    print("特征类别统计：",Counter(desTypeCount))
    print("特征类别数量：", len(set(desTypeCount)))
    print("______________4.模型训构建________________")
    AccTrain = []
    AccTest = []
    fucAll = []
    train_acc_allEpochs = []
    test_acc_allEpochs = []
    # 数据集切分
    data = pd.concat([dataY.iloc[:, -1], finalX], axis=1)
    # 依拉达准则清除异常数据
    data = CountErrorFeatureNum(pd.DataFrame(data))
    print(data.shape)
    XX, YY = splitData(data, type='r')
    print("________模型训练（PIC50回归）___________")
    modelPath = './modelAll'
    if not os.path.exists(os.path.join(modelPath,'model5.pkl')):
        # 模型训练
        rf, sfolder = modelRF(type='r', cv=k_folds, shuffle=True)
        trainACC, testACC = trainAndEvalModel(rf, sfolder, XX, YY, type='r')
        AccTrain.append(trainACC)
        AccTest.append(testACC)
        # 3.2 ADMET 5个特征训练
        labelsCount = []
        for i in range(5):
            print("特征" + str(i + 1) + "计算过程")
            print("________数据集划分___________")
            Y = dataY2.iloc[:, i + 1]
            data = pd.concat([Y, finalX], axis=1)
            # 依拉达准则清除异常数据
            data = CountErrorFeatureNum(pd.DataFrame(data))
            print(data.shape)
            # 正负样本统计
            labelsCount.append(list(Y.value_counts()))
            # 3.1 数据集切分
            XX, YY = splitData(data, type='c')
            # 模型训练
            print("________模型训练（ADMET分类）___________")
            rf, sfolder = modelRF(type='c', cv=k_folds, shuffle=True)
            trainACC, testACC = trainAndEvalModel(rf, sfolder, XX, YY, type='c')
            # 存储数据
            AccTrain.append(trainACC)
            AccTest.append(testACC)

        print("train model MSE/ACC:{} avg acc:{} +/- {}".format(AccTrain, np.mean(AccTrain[1:]), np.std(AccTrain[1:])))
        print("test model MSE/ACC:{} avg acc:{} +/- {}".format(AccTest, np.mean(AccTest[1:]), np.std(AccTest[1:])))
    print("______________6.智能优化算法多目标优化求问题求解_________________")
    # 数据准备
    features = finalX.to_numpy()
    features = StandardScaler().fit_transform(features)
    # print(features)
    # 加载模型
    for i in range(6):
        fucAll.append(torch.load(os.path.join(modelPath, 'model' + str(i)) + 'pkl'))
    # 最终目标函数
    n_vars = featuresNum
    n_objs = 6
    n_constrs = 1
    import autograd.numpy as anp
    xls = anp.array(list(features.min(axis=0)))
    xus = anp.array(list(features.max(axis=0)))
    MOPSolution(xls, xus, fucAll, n_vars, n_objs, n_constrs)


