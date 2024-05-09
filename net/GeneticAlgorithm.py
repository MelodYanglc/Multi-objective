import pygad.torchga
import pygad
import torch
from torch import nn

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, finalFC, loss_function
    """
    - model:模型
    - solution: 也就是遗传算法群体中的个体
    - data: 数据
    """
    predictions,fxNum,x  = pygad.torchga.predict(model=finalFC,solution=solution,data=data_inputs)
    # if fxNum.detach().numpy() < 3.0:
    #     return -1
    # else:
    #     pass
    # 计算误差
    abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001
    # 因为评估值是越大越好
    solution_fitness = 1.0 / abs_error
    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

def GA_algrithom(finalFC,data_inputs,num_generations,num_parents_mating,initial_population):
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           on_generation=callback_generation)
    ga_instance.run()
    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)
    # 返回最优参数的详细信息
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    # 基于最好的个体来进行预测
    predictions,fxNum,x = pygad.torchga.predict(model=finalFC, solution=solution, data=data_inputs)
    pred = predictions.detach().numpy()
    print("预测值 : \n", pred)
    print("ADMET对人体有利总数 : \n", fxNum)
    print("X值 : \n", x)
    print("pred shape:",pred.shape)


def GA_Params(model):
    # 在初始化种群时，实例化 pygad.torchga.TorchGA
    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=10)
    num_generations = 5# 迭代次数
    num_parents_mating = 5  # 每次从父类中选择的个体进行交叉、和突变的数量
    initial_population = torch_ga.population_weights  # 初始化网络权重
    return num_generations,num_parents_mating,initial_population


class finalAimFunction(nn.Module):
    def __init__(self, fucAll):
        super(finalAimFunction, self).__init__()
        self.gx = fucAll[0]
        self.f1 = fucAll[1]
        self.f2 = fucAll[1]
        self.f3 = fucAll[1]
        self.f4 = fucAll[1]
        self.f5 = fucAll[1]

    def forward(self, x):
        out = 0.53505669 * self.gx(x)
        _, predicted1 = torch.max(self.f1(x).data, 1)
        out =out + 0.24292175 * predicted1.unsqueeze(1)
        _, predicted2 = torch.max(self.f2(x).data, 1)
        out =out +  0.74377414 * predicted2.unsqueeze(1)
        _, predicted3 = torch.max(self.f3(x).data, 1)
        out = out + 0.3388965 * predicted3.unsqueeze(1)
        _, predicted4 = torch.max(self.f4(x).data, 1)
        out = out + 0.17461334 * predicted4.unsqueeze(1)
        _, predicted5 = torch.max(self.f5(x).data, 1)
        out = out + 0.14757196 * predicted5.unsqueeze(1)
        fxNum = predicted1.unsqueeze(1) + predicted2.unsqueeze(1) +\
                predicted3.unsqueeze(1) + predicted4.unsqueeze(1) + predicted5.unsqueeze(1)

        return out,fxNum,x