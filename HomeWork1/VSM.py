# coding:utf-8
#author：liubing
import os
import re                                               
from nltk.corpus import stopwords as sw
from textblob import *
import math
from collections import Counter
import time

def Txt2wordLst(txt):                                                       #对单个文本进行处理：转为小写字母、分割、去掉无用的词、词形还原、去掉带数字的、去停用词
    wordLst = re.findall("\w+",str.lower(txt))
    wordLst = [word for word in wordLst if len(word) >= 3]                  #去掉小于3个字母的词    
    wordLst=WordList(wordLst).singularize()                                 #名词变单数
    wordLst=[word.lemmatize('v') for word in wordLst]                       #动词的过去式、进行时变一般形式     
    stopWords=sw.words("english")
    wordLst=[word for word in wordLst if word.isalpha()]                    #去带数字的   
    wordLst=[word for word in wordLst if word not in stopWords]             #去停用词
    return wordLst

def txtReduce(wrdLst):   #去除词频低的词
    dic=dict(Counter(wrdLst))
    halfFreq=max(dic.values()) *0.5 #最大词频的0.5倍    
    dic1={}    
    for key, value in dic.items():
        if value > 3 or value >halfFreq:
            dic1[key]=value            
    return dic1

def cal_tfidf(tf,df,docNum):
    if tf>0:
        tf=1+math.log(tf)
    else:
        tf=0
    idf=math.log(docNum/df)        
    tfidf=tf*idf        
    return tfidf
    
def Tf_idfLst(dsLst):
    docCnt = 0
    tfIdfLst=[]    
    wdLst=[]
    topicLst=[]    
    tmp=[]
    for tpLst in dsLst:
        for wdDic in tpLst:
            docCnt+=1
            lst=list(wdDic)
            tmp.extend(lst) 
    dfDic=dict(Counter(tmp))
    dicLst=list(dfDic) 
       
    n=0    
    f=open("vsmResult.txt","a")
    for tpLst in dsLst:
        for wdDic in tpLst:
            for wd in dicLst:
                if wdDic.__contains__(wd):
                    tf=wdDic[wd]
                    df=dfDic[wd]
                    tfidf=cal_tfidf(tf,df,docCnt)
                else :
                    tfidf=0                                                
                wdLst.append(tfidf)                
            tem1=wdLst.copy()
            topicLst.append(tem1) 
            wdLst.clear()
            n+=1             
        tmp2=topicLst.copy()
        tfIdfLst.append(tmp2)
        topicLst.clear()
        print('正在保存' ) 
        f.write(str(tmp2))    
    print("保存完毕")     
    f.close()      
    return tfIdfLst,dicLst

def DataSource2TfIdfLst(dataSourcePath):
    n=0
    docCnt=0#选80%文档间词典
    dsLst=[]    #所有数据集
    topicLst=[] #一类数据集   
    tmp=[]     
    foldersLst=os.listdir(dataSourcePath)                                       #所有topic文件夹的列表
    for topicFolder in foldersLst:  
        filesLst = os.listdir(dataSourcePath+topicFolder)
        path=dataSourcePath+topicFolder
        totalNum=len([name for name in os.listdir(path)])
        for fileName in filesLst:
            docCnt+=1                          
            filepath = dataSourcePath+topicFolder+'\\'+fileName            
            txt = open(filepath,'rb').read().decode('GBK', 'ignore')
            wrdLst=Txt2wordLst(txt)                                            #对文本进行预处理：大小写、非英语内容、分词、词型转换、字母、去停            
            wrdDic=txtReduce(wrdLst)    #单个文档的词汇及其词频构成的一个字典
            topicLst.append(wrdDic)   #所有的词典构成的列表
            n+=1
            print('正在预处理第： ',n,' 个文档')
            if docCnt>totalNum*0.8: #处理60%的文档docCnt=totalNum*0.6
                docCnt=0
                break            
        tmp=topicLst.copy()     
        dsLst.append(tmp)
        topicLst.clear()                 
    tf_idfLst,dic=Tf_idfLst(dsLst)#所有词典构成列表，key为词，value为tfidf值
    return tf_idfLst,dic,dsLst

if __name__=="__main__":
    dataSourcePath=os.getcwd()+"\\20news-18828\\"                               #数据集的存放路径
    t1=time.time()
    DataSource2TfIdfLst(dataSourcePath)
    t2=time.time()
    print("总耗时为",t2-t1,"秒")

