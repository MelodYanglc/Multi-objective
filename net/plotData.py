import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plotLabelsCount(labelsCount):
    plt.clf()
    # print(labelsCount)
    x = np.arange(2)
    a = 0.05
    plt.bar(x - 2 * a, labelsCount[0], width=0.05)
    plt.bar(x - a, labelsCount[1], width=0.05)
    plt.bar(x, labelsCount[2], width=0.05)
    plt.bar(x + a, labelsCount[3], width=0.05)
    plt.bar(x + 2 * a, labelsCount[4], width=0.05)
    plt.xlabel(" Labels")
    plt.ylabel("Numbers")
    plt.legend(['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN'])
    plt.xticks([0,1])
    plt.show()

def plotModelLoss(train_acc_allEpochs_path,test_acc_allEpochs_path):
    dataTrain = pd.read_csv(train_acc_allEpochs_path)
    dataTrain = dataTrain.iloc[:, 1:]
    print("训练集误差/准确度：", dataTrain)
    dataTest = pd.read_csv(test_acc_allEpochs_path)
    dataTest = dataTest.iloc[:,1:]
    print("测试集误差/准确度：",dataTest)
    plt.clf()
    features = ['PIC50','Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
    colors = ['r','g','b','c','m','y']
    for i in range(6):
        plt.subplot(2,3,(i+1))
        x = range(len(dataTest.iloc[i,:]))
        plt.plot(x, dataTrain.iloc[i, :])
        plt.plot(x,dataTest.iloc[i,:],color = colors[i])
        if i == 0:
            plt.ylabel('Mean square error')
        else:
            plt.ylabel('Accuracy')
        plt.xlabel("Epochs")
        plt.grid(linestyle='-.')
        plt.legend([features[i] + ' Train',features[i] + ' Test'])
    plt.show()

