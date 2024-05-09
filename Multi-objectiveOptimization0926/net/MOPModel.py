import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import autograd.numpy as anp
from pymoo.core.problem import Problem
# AGEMOEA算法求解
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize

from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.algorithms.moo.nsga2 import NSGA2  # 最新版已经发生改变
from pymoo.algorithms.moo.rnsga3 import RNSGA3, NSGA3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.soo.nonconvex.pso import PSO


# 采样方法
from pymoo.operators.sampling.rnd import FloatRandomSampling,PermutationRandomSampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
# 选择方法
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.selection.rnd import RandomSelection
# 交叉方法
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.spx import SPX
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.crossover.pcx import PCX
from pymoo.operators.crossover.erx import EdgeRecombinationCrossover
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.crossover.nox import NoCrossover
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.crossover.ox import OrderCrossover
# 变异方法
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.mutation.bitflip import BinaryBitflipMutation
from pymoo.operators.mutation.pm import PM

from pymoo.visualization.pcp import PCP
from sklearn.preprocessing import StandardScaler,MinMaxScaler

device = "cuda"#torch.device('cpu')

class MyProblem(Problem):
    def __init__(self,
                 xl,  # 变量下界
                 xu,  # 变量上界
                 fucAll,
                 n_var,  # 变量数
                 n_obj,  # 目标数
                 n_constr,  # 约束数
                 ):
        super().__init__(n_var=n_var,n_obj=n_obj,n_constr=n_constr,xl=xl,xu=xu)
        self.fucAll = fucAll

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Tensor(x).to(device)
        # 定义目标函数
        h1 = self.fucAll[0](x)
        _, h2 = torch.max(self.fucAll[1](x).data, 1)
        _, h3 = torch.max(self.fucAll[2](x).data, 1)
        _, h4 = torch.max(self.fucAll[3](x).data, 1)
        _, h5 = torch.max(self.fucAll[4](x).data, 1)
        _, h6 = torch.max(self.fucAll[5](x).data, 1)
        # 定义约束条件
        g1 = 3 - (h2.unsqueeze(1) + h3.unsqueeze(1) +\
                h4.unsqueeze(1) + h5.unsqueeze(1) + h6.unsqueeze(1))

        # todo
        out["F"] = anp.column_stack([-h1.detach().cpu().numpy(), -h2.detach().cpu().numpy(), -h3.detach().cpu().numpy(),
                                     -h4.detach().cpu().numpy(), -h5.detach().cpu().numpy(), -h6.detach().cpu().numpy()])
        out["G"] = anp.column_stack([g1.detach().cpu().numpy()])

def NSGA2Model():
    # 定义遗传算法
    algorithm = NSGA2(
        pop_size=200,
    )
    return algorithm

def NSGA3Model():
    algorithm = NSGA3(
        pop_size=200,
        ref_dirs=get_reference_directions("das-dennis", 6, n_partitions=12)
    )
    return algorithm

# def AGEMOEA2Model():
#     algorithm = AGEMOEA2(
#         pop_size=200,
#      )
#     return algorithm

def AGEMOEAModel():
    algorithm = AGEMOEA(
        pop_size=200,
        sampling=FloatRandomSampling(),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(prob=None, eta=20),
     )
    return algorithm

def ImporveAGEMOEA2Model():
    algorithm = AGEMOEA(pop_size=200,
                        sampling=FloatRandomSampling(),
                         selection=TournamentSelection(func_comp=binary_tournament),
                         # crossover=HalfUniformCrossover(),
                        crossover=DEX(),
                         mutation=PM(eta=20),
                        )
    return algorithm

def ModelSolution(finalFC,algorithm,nGen):
    # 求解方程
    res = minimize(finalFC,
                   algorithm,
                   ('n_gen', nGen),
                   seed=1,
                   save_history=True,
                   verbose=True,
                   )
    return res

def ModelSolution2(finalFC,algorithm,nGen):
    # 求解方程
    res = minimize(finalFC,
               algorithm,
               ('n_gen', nGen),
               seed=1,
               verbose=False)
    return res

def plotAll(res):
    # print(np.max(abs(res.F)))
    # 非劣解集
    # min_F = np.min(np.sum(res.history[-1].pop.get("F"), axis=1))
    # print(min_F)
    # for e in res.history:
    #     print("------")
    #     if np.min(np.sum(e.pop.get("F"), axis=1)) == min_F:
    #         print(type(e.pop.get("X")))
    #         print(e.pop.get("X").shape)
    #         print(e.pop.get("F"))
    #         print(np.where(np.min(np.sum(e.pop.get("F"), axis=1)) == min_F))
    # ret = [e.pop.get("X")[0] for e in res.history if np.min(np.sum(e.pop.get("F"),axis=1)) == min_F]
    # print("ret x:", len(ret))
    # print("ret x 0:", ret[0])
    # print("ret x 1:", ret[1])
    # 求解过程可视化，折线图，散点图
    val = np.array([-e.opt.get("F")[0] for e in res.history])
    legends = ['PIC50', 'Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(6):
        plt.subplot(2, 3, (i + 1))
        plt.plot(range(len(val)), val[:, i], color=colors[i])
        plt.ylabel('Value')
        plt.xlabel("Epochs")
        if i >= 1:
            plt.ylim(-0.25, 1.25)
        plt.grid(linestyle='-.')
        plt.legend([legends[i]])
    plt.show()

    # 求解结果可视化，雷达图
    # min_r = 0
    # max_r = int(np.max(abs(res.F)))
    # sacle_r = int((2/3)*max_r)
    # from pymoo.visualization.petal import Petal
    # plot2 =Petal(bounds=[min_r, max_r],
    #       cmap="tab20",
    #       labels=['PIC50','Caco-2','CYP3A4','hERG','HOB','MN']
    #       )
    # plot2.add(abs(res.F[0,:])*[1,sacle_r,sacle_r,sacle_r,sacle_r,sacle_r]).show()
    #折线图
    plot2PCP(res)
    #雷达图
    plotLD(res)


def plot2PCP(res):
    MOP_array = res.to_numpy()
    # MOP_array = MinMaxScaler().fit_transform(MOP_array)
    print("MOP_array:",MOP_array)
    min_r = 0  # np.min(abs(res.F[:,0]))
    max_r = 1  # int(np.max(abs(res.F))) + 1
    pl = PCP(labels=['PIC50', 'Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN'])
    pl.set_axis_style(color="grey", alpha=0.5)
    pl.add(MOP_array, color="grey", alpha=0.3)
    pl.add(MOP_array[-2, :], linewidth=3, color="red")
    pl.normalize_each_axis = False
    pl.bounds = [[min_r, 0, 0, 0, 0, 0], [max_r, 1, 1, 1, 1, 1]]
    pl.show()

def plotPC(df):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from pandas.plotting import parallel_coordinates
    fig, axes = plt.subplots()
    columns = df.columns
    dataS = MinMaxScaler().fit_transform(df)
    scaleWeigth = np.linspace(0.9,1.1,dataS.shape[0])
    FFF = np.tile(scaleWeigth, (dataS.shape[1], 1))
    FFF = np.transpose(FFF)
    data = np.multiply(FFF,dataS)
    # print(np.tile(scaleWeigth, (, 2)).shape)
    # print(data.shape)
    # print(dataF.shape)
    data = pd.DataFrame(data,columns= columns)
    data['ss'] = range(data.shape[0])
    parallel_coordinates(data, 'ss', ax=axes)
    plt.grid(False)
    axes.legend_.remove()
    plt.show()


def plotLD(res):
    res = res.to_numpy()
    angles = np.linspace(0, 2 * np.pi, len(res[0, :]), endpoint=False)
    # print(angles)
    colorsL = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    labelsL = ['PIC50', 'Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
    # 设置图形的大小
    fig = plt.figure(figsize=(8, 6), dpi=100)
    # 新建一个子图
    for i in range(4):
        # 尺寸放缩
        min_r = 0
        max_r = int(np.max(abs(res[i, :])))
        sacle_r = int((2 / 3) * max_r)
        sacleL = [1, sacle_r, sacle_r, sacle_r, sacle_r, sacle_r]
        # 使雷达图数据封闭
        sacleLIN = np.concatenate((sacleL, [sacleL[0]]))
        dataIN = np.concatenate((abs(res[i, :]), [abs(res[i, 0])]))
        anglesIN = np.concatenate((angles, [angles[0]]))
        labelsIN = np.concatenate((labelsL, [labelsL[0]]))
        ax = plt.subplot(2, 2, (i + 1), polar=True)
        # print("dataIn:",dataIN)
        # print("scaleIn:",sacleLIN)
        # print("mult:",dataIN*sacleLIN)
        ax.plot(anglesIN, dataIN * sacleLIN, 'o-', color=colorsL[i])
        # 设置雷达图中每一项的标签显示
        ax.set_thetagrids(anglesIN * 180 / np.pi, labelsIN)
        # 设置雷达图的0度起始位置
        ax.set_theta_zero_location('N')
        plt.grid(c='gray', linestyle='--', )
    plt.show()

def plotCorr(res):
    MOP_result = res
    print(MOP_result.corr())
    sns.heatmap(MOP_result.corr(), annot=True, linewidths=.5, cmap="YlGnBu")
    plt.show()


def MOPSolution(xls,xus,fucAll,n_vars,n_objs,n_constrs):
    # 超参数设置
    FlodNum = 2
    genNum = 50
    # 优化目标定义
    finalFC = MyProblem(xl=xls,  # 变量下界
                        xu=xus,  # 变量上界
                        fucAll=fucAll,
                        n_var=n_vars,  # 变量数
                        n_obj=n_objs,  # 目标数
                        n_constr=n_constrs  # 约束数
                        )

    savePath = './MOPParetoFront0910/ParetoFrontAll.csv'
    if  os.path.exists(savePath):
        # 取消警告
        from pymoo.config import Config
        Config.show_compile_hint = False
        # print("xls:",xls)
        # print("xus:", xus)
        print("--------计算MOP帕累托前端-------------")
        resAll = []
        res1All = []
        res2All = []
        res3All = []
        res4All = []
        res5All = []
        for i in range(FlodNum):
            modelNSGA2 = NSGA2Model()
            modelNSGA3 = NSGA3Model()
            modelAGEMOEA = AGEMOEAModel()
            # modelAGEMOEA2 = AGEMOEA2Model()
            modelImporveAGEMOEA2 = ImporveAGEMOEA2Model()

            res5 = ModelSolution(finalFC, modelImporveAGEMOEA2, genNum)
            res1 = ModelSolution(finalFC, modelNSGA2,genNum)
            res2 = ModelSolution(finalFC, modelNSGA3,genNum)
            res3 = ModelSolution(finalFC, modelAGEMOEA,genNum)
            # res4 = ModelSolution2(finalFC, modelAGEMOEA2,genNum)


            # print(np.array(abs(res1.F)).shape,np.array(abs(res2.F)).shape,np.array(abs(res3.F)).shape,np.array(abs(res4.F)).shape)
            # 可视化
            # plotAll(res4)
            res1All.append(abs(res1.F))
            res2All.append(abs(res2.F))
            res3All.append(abs(res3.F))
            # res4All.append(abs(res4.F))
            res5All.append(abs(res5.F))
            resF = np.r_[(abs(res1.F), abs(res2.F),abs(res3.F),abs(res5.F))]
            resAll.append(resF)

        Farray = resAll[0]
        for i in range(1,len(resAll)):
            Farray = np.r_[Farray,resAll[i]]
        # 去除重复值，保存为csv文件
        FarrayDropDuplicates = pd.DataFrame(Farray,columns=['PIC50', 'Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN'])\
            .drop_duplicates(subset=None, keep='first', inplace=False)
        # print("FarrayDropDuplicates:",FarrayDropDuplicates.shape)
        FarrayDropDuplicates.to_csv(savePath)
    #加载ParetoFront数据
    FarrayDropDuplicates = pd.read_csv(savePath)
    #绘制图形
    print("-----------绘制图形，分析各项目标函数之间的冲突关系-------------")
    # plotPC(FarrayDropDuplicates)
    # plot2PCP(FarrayDropDuplicates)
    # plotCorr(FarrayDropDuplicates)
    # plotLD(FarrayDropDuplicates)
    # print("Farray:", Farray)
    # print("Farray:", Farray.shape)
    # print("Farray df:", FarrayDropDuplicates)
    # print("Farray df:", FarrayDropDuplicates.shape)
    # print(res.X)
    # print(abs(res.F))  # 显示结果
    print("-----------计算各个模型的预测结果-------------")
    MOPSoultionPath = './MOPParetoSolutionResult0910'
    if  os.path.exists(os.path.join(MOPSoultionPath,'MOPresult3.csv')):
        modelNSGA2 = NSGA2Model()
        modelNSGA3 = NSGA3Model()
        modelAGEMOEA = AGEMOEAModel()
        # modelAGEMOEA2 = AGEMOEA2Model()
        modelImporveAGEMOEA2 = ImporveAGEMOEA2Model()
        modelAll = [modelNSGA2,modelNSGA3,modelAGEMOEA,modelImporveAGEMOEA2]
        for i in range(len(modelAll)):
            result = []
            for j in range(FlodNum):
                res = ModelSolution(finalFC, modelAll[i], genNum)
                result.append(abs(res.F))
                #####################################
                # print("-------------算法收敛性分析---------------")
                # from pymoo.util.running_metric import RunningMetric
                # running = RunningMetric(delta_gen=100,
                #                         n_plots=5,
                #                         only_if_n_plots=True,
                #                         key_press=False,
                #                         do_show=True)
                # for algorithm in res.history:
                #     running.notify(algorithm)
                #####################################
                # hist = res.history
                # # print(len(hist))  # 40
                #
                # n_evals = []  # corresponding number of function evaluations\
                # hist_F = []  # the objective space values in each generation
                # hist_cv = []  # constraint violation in each generation
                # hist_cv_avg = []  # average constraint violation in the whole population
                #
                # for algo in hist:
                #     # store the number of function evaluations
                #     n_evals.append(algo.evaluator.n_eval)
                #     # retrieve the optimum from the algorithm
                #     opt = algo.opt
                #     # store the least contraint violation and the average in each population
                #     hist_cv.append(opt.get("CV").min())
                #     hist_cv_avg.append(algo.pop.get("CV").mean())
                #     # filter out only the feasible and append and objective space values
                #     feas = np.where(opt.get("feasible"))[0]
                #     hist_F.append(opt.get("F")[feas])

                #############################################
            resultConcat = result[0]
            for k in range(1, len(result)):
                resultConcat = np.r_[resultConcat, result[k]]
            pd.DataFrame(resultConcat).to_csv(os.path.join(MOPSoultionPath,'MOPresult'+str(i)+'.csv'))



    print("-----------评估各个模型的性能-------------")
    # 加载MOP求解结果，评估各项算法的性能
    from pymoo.indicators.igd_plus import IGDPlus
    from pymoo.indicators.hv import Hypervolume

    for i in range(4):
        resultF = pd.read_csv(os.path.join(MOPSoultionPath,'MOPresult'+str(i)+'.csv'))
        print(FarrayDropDuplicates.to_numpy().shape)
        print(np.array(resultF).shape)
        refF1 = FarrayDropDuplicates.to_numpy()
        ind = IGDPlus(refF1)
        predF1 = np.array(resultF)
        evaR1 = ind.do(predF1)
        print("model" +str(i)+",IGD+:", evaR1)

        ref_point = FarrayDropDuplicates.to_numpy().max(axis=0) + 1
        # print(ref_point.shape)
        ind = Hypervolume(ref_point=ref_point)
        predF2 = np.array(resultF)
        evaR2 = ind.do(predF2)
        print("model" +str(i)+",HV:", evaR2)





if __name__ =="__main__":
    # plotPC()
    ssC = np.linspace(0.8,1.2,18)
    FFF = np.tile(ssC,(20,1))
    FFF = np.transpose(FFF)
    print(FFF.shape)






