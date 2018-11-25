# coding:utf-8
# author:liubing
from numpy import *
import json


def StoreTrainResult():
    #     vocabList, nbTrainVecLst, nbClsRatLst, feature = trainNB(dataSourcePath)
    list_log = ones(3)
    print("list_log", list_log)
    list_log = json.dumps(list_log.tolist())
    file = open('log.txt', 'w')
    for fp in list_log:
        file.write(str(fp))
    file.close()

    return True


def LoadTrainResult():
    StoreTrainResult()
    file = open('log.txt', 'r')
    list_read = file.read()
    list_read = json.loads(list_read)
    list_read = array(list_read)
    return list_read


print(LoadTrainResult())
