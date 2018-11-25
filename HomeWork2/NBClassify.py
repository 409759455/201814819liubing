# coding:utf-8
# author:liubing

import os
import re
from textblob import *
from nltk.corpus import stopwords as sw
from collections import Counter
import time
from numpy import *
import json


def Txt2wordLst(txt):  # 对单个文本进行处理：转为小写字母、分割、去掉无用的词、词形还原、去掉带数字的、去停用词
    wordLst = re.findall("\w+", str.lower(txt))
    wordLst = [word for word in wordLst if len(word) >= 3]         # 去掉小于3个字母的词
    wordLst = WordList(wordLst).singularize()                       # 名词变单数
    wordLst = [word.lemmatize('v')for word in wordLst]        # 动词的过去式、进行时变一般形式
    stopWords = sw.words("english")
    wordLst = [word for word in wordLst if word.isalpha()]          # 去带数字的
    wordLst = [word for word in wordLst if word not in stopWords]   # 去停用词
    return wordLst


def TxtReduce(wrdLst):                                              # 去除词频低的词
    dic = dict(Counter(wrdLst))
    halfFreq = int(max(dic.values()) * 0.5)                         # 最大词频的0.5倍
    wdlst = []
    for key, value in dic.items():
        if value > 2 or value > halfFreq:
            for i in range(0, value):                                          # 保留重复的词
                wdlst.append(key)
    return wdlst


def loadDataSet(dataSourcePath, part):  # part=0加载前80%文档作为训练文档，part=1加载后20%文档作为测试文档
    debug1, debug2 = 0, 0  # 调试用
    postingList = []                                # 所选文档构成的list
    docCntLst = []                                  # 每类文档的个数
    categories = os.listdir(dataSourcePath)         # 所有category文件夹的列表
    for category in categories:
        catPath = dataSourcePath + category
        catDocs = os.listdir(catPath)
        totalNum = len([name for name in os.listdir(catPath)])  # 某一类的文档总数
        docNum = int(totalNum * 0.8)
        if part == 0:                               # part=0为前80%文档
            docCntLst.append(docNum)                # 记录各类文档的数量
        else:                                       # part=1为后20%文档
            docCntLst.append(totalNum - docNum)     # 记录各类文档的数量
        docCnt = 0
        for catDoc in catDocs:
            if (part == 0) and (docCnt < docNum):                             # part=0返回前80%文档作为训练文档
                docPath = dataSourcePath + category + '\\' + catDoc
                doc = open(docPath, 'rb').read().decode('GBK', 'ignore')
                wrdLst1 = Txt2wordLst(doc)  # 对文本进行预处理：大小写、非英语内容、分词、词型转换、字母、去停
                wrdLst = TxtReduce(wrdLst1)
                postingList.append(wrdLst)
                debug1 += 1
                print("加载第", debug1, "个训练文档")
                print("训练文档路径为", docPath)
            # >= docNum                       # part=1返回后20%文档作为测试文档
            if (part == 1) and (docCnt >= docNum):
                docPath = dataSourcePath + category + '\\' + catDoc
                doc = open(docPath, 'rb').read().decode('GBK', 'ignore')
                # 对文本进行预处理：大小写、非英语内容、分词、词型转换、字母、去停
                wrdLst1 = Txt2wordLst(doc)
                wrdLst = TxtReduce(wrdLst1)
                postingList.append(wrdLst)
                debug2 += 1
                print("加载第", debug2, "个测试文档")
                print("测试路径为", docPath)
            docCnt += 1
    return postingList, docCntLst


def setOfWords2VecLst(vocabList, inputSet):            # 文档变向量 和字典一样长，词条出现与否为0，1
    returnVecLst = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:  # 当前文档中有某个词条，则根据词典获取其位置并赋值1
            returnVecLst[vocabList.index(word)] = 1
    return returnVecLst


def bagOfWords2Vec(vocabList, inputSet):            # 文档变成向量，长度和字典一样，表示各词在文档中出现的总次数
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def createVocabList(dataSet):       # 构建词典
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 所有词
    print("字典构建完成,字典单词数目为", len(list(vocabSet)))
    return list(vocabSet)


def trainNB(dataSourcePath):
    bagVecLst = []                  # 文档词典的bag表示
    nbClsRatLst = []                # 各类占的比率
    nbVecLst = []                   # 各类中的文档的nb向量列表
    postingList, docCntLst = loadDataSet(
        dataSourcePath, 0)  # 参数2为0表示获取数据集中前80%文档
    vocabList = createVocabList(postingList)
    i = 0
    for doc in postingList:
        bagVecLst.append(bagOfWords2Vec(vocabList, doc))
        print("bagVecLst: ", i)
        i = i + 1
    sumVec = ones([20, len(vocabList)])  # 带平滑处理了
    for vec in bagVecLst:
        sumVec += vec
    # 各特征的概率的对数的和(用作test时算概率时的分母)
    features = sum(log(sumVec / len(bagVecLst)))
    s = 0                                           # 分类的起始位置
    catBagVecSum = ones(len(vocabList))
    for cnt in docCntLst:                           # 按文档数划分类
        catBagVecLst = bagVecLst[s:(s + cnt)]
        s += cnt
        catWrdSum = sum(catBagVecLst)
        for vecbag in catBagVecLst:
            catBagVecSum += vecbag
        nbVec = log(catBagVecSum / catWrdSum)
        nbVecLst.append(nbVec.tolist())
        clsRat = cnt / sum(docCntLst)
        nbClsRatLst.append(clsRat)                  # 各类文档占比
    print("nbVecLst", nbVecLst)
    print("len（nbVecLst）", len(nbVecLst))
#     debug1 = input("stop@125 line")
    return vocabList, nbVecLst, nbClsRatLst, features


def StoreTrainResult():  # 将训练集结果和测试文档list保存到硬盘，便于多次运行测试集调试时节约运行时间
    vocabList, nbTrainVecLst, nbClsRatLst, feature = trainNB(dataSourcePath)
    # 第二个参数为1表示获取第一个参数路径下每个文件夹内的后20%文档
    postingList, docCntLst = loadDataSet(dataSourcePath, 1)
    list_log = json.dumps(vocabList)
    file = open('vocabList.txt', 'w')
    for fp in list_log:
        file.write(str(fp))
    file.close()
    print("vocabList.txt保存完成！")

    list_log = json.dumps(nbTrainVecLst)
    file = open('nbTrainVecLst.txt', 'w')
    for fp in list_log:
        file.write(str(fp))
    file.close()
    print("nbTrainVecLst.txt保存完成！")

    list_log = json.dumps(nbClsRatLst)
    file = open('nbClsRatLst.txt', 'w')
    for fp in list_log:
        file.write(str(fp))
    file.close()
    print("nbClsRatLst.txt保存完成！")

    list_log = json.dumps(feature.tolist())
    file = open('feature.txt', 'w')
    for fp in list_log:
        file.write(str(fp))
    file.close()
    print("feature.txt保存完成！")

    list_log = json.dumps(postingList)
    file = open('posting2List.txt', 'w')
    for fp in list_log:
        file.write(str(fp))
    file.close()
    print("posting2List.txt保存完成！")

    list_log = json.dumps(docCntLst)
    file = open('docCnt2Lst.txt', 'w')
    for fp in list_log:
        file.write(str(fp))
    file.close()
    print("docCnt2Lst.txt保存完成！")
    return True


def LoadTrainResult():
    file = open('vocabList.txt', 'r')
    list_read = file.read()
    vocabList = json.loads(list_read)
    file.close()
    file = open('nbTrainVecLst.txt', 'r')
    list_read = file.read()
    nbTrainVecLst = json.loads(list_read)
    file.close()
    file = open('nbClsRatLst.txt', 'r')
    list_read = file.read()
    nbClsRatLst = json.loads(list_read)
    file.close()

    file = open('feature.txt', 'r')
    list_read = file.read()
    feature = json.loads(list_read)
    file.close()
    print("从硬盘加载训练参数，加载完成！")
    return vocabList, nbTrainVecLst, nbClsRatLst, feature


def LoadTestDocLst():
    file = open('posting2List.txt', 'r')
    list_read = file.read()
    postingList = json.loads(list_read)
    file.close()

    file = open('docCnt2Lst.txt', 'r')
    list_read = file.read()
    docCntLst = json.loads(list_read)
    file.close()
    print("从硬盘加载测试集，加载完成！")

    return postingList, docCntLst


# 返回testDocVec文档的类别代号（ 从1-20共20个类。）
def classifyNB(testDocVecLst, nbTrainVecLst, nbClsRatLst, feature):
    catNo = []
    for (nbVec, nbCls) in zip(nbTrainVecLst, nbClsRatLst):
        print("sum(testDocVecLst)", sum(testDocVecLst))
        nb = sum(testDocVecLst * array(nbVec)) + log(nbCls)
        prob = sum(array(nbVec)) + log(nbCls)
        nb = nb / prob
        print("testNb", nb)
        catNo.append(nb)
    number = catNo.index(max(catNo)) + 1
    print("上一文档属于第", number, "类")
    return number


def testingNB(dataSourcePath):
    debug = 0
    debug1 = 0
    rightCnt = 0                # 分类正确的计数值
    wrongCnt = 0                # 分类错误的计数值
    nbTestVecLst = []           # 所有测试文档的nb向量列表
    catNo = 0                   # 类别代号，从1-20，分别代表20个类
    # 根据训练集计算结果返回：词典、各类词频向量、各类的先验概率list。
    vocabList, nbTrainVecLst, nbClsRatLst, feature = LoadTrainResult()
    # 以下表示返回数据集中后20%文档，返回值中postingList为所有文档的list,docCntLst为各类文档总数
    postingList, docCntLst = LoadTestDocLst()
    for doc in postingList:     # 将测试文档以词典向量方式表示
        nbTestVecLst.append(bagOfWords2Vec(vocabList, doc))
    #*************
#     list_log = json.dumps(nbTestVecLst)
#     file = open('nbTestVecLst.txt', 'w')
#     for fp in list_log:
#         file.write(str(fp))
#     file.close()
#     #******************
#     file = open('nbTestVecLst.txt', 'r')
#     list_read = file.read()
#     nbTestVecLst = json.loads(list_read)
#     file.close()
#     print("从硬盘加载nbTestVecLst，加载完成！")
    #*********************

    s = 0
    for cnt in docCntLst:       # 可以根据文档个数区分类别（每类有20%文档，共同构成一个list）
        catNo += 1
        catVecLst = nbTestVecLst[s:(s + cnt)]
        s += cnt
        for doc in catVecLst:
            if classifyNB(doc, nbTrainVecLst, nbClsRatLst, feature) == catNo:
                rightCnt += 1
            else:
                wrongCnt += 1
            debug += 1
            print("完成了第", debug, "个测试文档分类的验证！")
    if(rightCnt + wrongCnt != 0):  # 计算分类正确率
        rRate = rightCnt / (rightCnt + wrongCnt)
    else:
        rRate = 0
    return rRate


if __name__ == '__main__':
    t1 = time.time()
    print("开始...")
    dataSourcePath = os.getcwd() + "\\20news-18828\\"           # 数据集的存放路径
#     StoreTrainResult()  # 第一次时运行，将训练集结果保存到硬盘，便于多次运行测试集调试时节约运行时间
    print("测试的正确率为：", testingNB(dataSourcePath) * 100, "%")
    t2 = time.time()
    print("总耗时为", t2 - t1, "秒")
