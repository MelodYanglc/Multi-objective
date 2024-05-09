import pandas as pd

def loadData(path,sheetName):
    data = pd.read_excel(path,sheetName)
    return data