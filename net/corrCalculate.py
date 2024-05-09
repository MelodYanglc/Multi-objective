import pandas as pd
import numpy as np

def corrData(data,choiceNum):
    res = np.corrcoef(np.array(data.iloc[:, 1:]).T, np.array(data.iloc[:, 0]).T)
    print(res.shape)
    #绝对相关度
    res = pd.DataFrame(abs(res[0, :]), index=data.columns, columns=['relative'])
    print(res)
    res = res.dropna()
    print(res.sort_values(by='relative'))
    result = res.sort_values(by='relative').iloc[-(choiceNum+1):, :]
    indexsF = result.index[:-1]
    print(indexsF)
    return indexsF