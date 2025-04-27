import codecs
import time
import numpy as np
import pandas as pd

root_path=''

result_path='/Result'

#加载apriori所需数据
def loadData(inFile):
    dataSet = []
    for line in inFile:
        line = line.strip().split(',')
        dataLine = [int(word) for word in line if word.isdigit()]
        dataSet.append(dataLine)
    return dataSet