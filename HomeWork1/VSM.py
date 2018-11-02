# coding:utf-8
#作者：liubing
import os
import re                                               
from nltk.corpus import stopwords as sw
from textblob import *
import math
import operator

def Txt2wordLst(txt):                                                          #对单个文本进行处理：转为小写字母、分割、去掉无用的词、词形还原、去掉带数字的、去停用词
    wordLst = re.findall("\w+",str.lower(txt))
    wordLst = [word for word in wordLst if len(word) >= 3]                  #去掉小于3个字母的词    
    wordLst=WordList(wordLst).singularize()                                 #名词变单数
    wordLst=[word.lemmatize('v') for word in wordLst]                       #动词的过去式、进行时变一般形式     
    stopWords=sw.words("english")
    wordLst=[word for word in wordLst if word.isalpha()]                    #去带数字的   
    wordLst=[word for word in wordLst if word not in stopWords]             #去停用词
    return wordLst

def txt2Dic(txt):    
    dic={}
    dic1={}
    for word in txt:
        if(dic.__contains__(word)):
            dic[word]=dic[word]+1
        else:
            dic[word]=1 
        print(word)
    halfFreq=max(dic.values()) *0.5 #最大词频的0.5倍
    print("词频的一半" ,halfFreq,"词频的一半")
    for key, value in dic.items():
        if value > 3 or value >halfFreq:
            dic1[key]=value        
    return dic1

def dfCal(key,DfDicLst):
    df=0
    for dic in DfDicLst:
        if dic.__contains__(key) :
            df+=1;
    return df

def cal_tfidf(tf,df,docNum):
    if tf>0:
        tf=1+math.log(tf)
    else:
        tf=0
    idf=math.log(docNum/df)        
    tfidf=tf*idf        
    return tfidf
    
def Tf_idfDicLst(tfDicLst):
    i=0#调试用
    j=0
    dicall={}
    f=open("vsmResult.txt","a")
    tfidfDicLst=tfDicLst.copy()
    docNum=len(tfidfDicLst)#值变为TfIdf
    for dic in tfidfDicLst:
        i+=1
        
        f.write("\n")
        f.write("\n")
        f.write("以下是文档")
        f.write(str(i))
        f.write("的Tf_Idf值")
        f.write("\n")
        
        for key in dic:
            j+=1
            df=dfCal(key,tfDicLst)
            tf=dic[key]            
            dic[key]=cal_tfidf(tf,df,docNum)            
            f.write(key)
            f.write(':')
            f.write(str(dic[key]))
            f.write("\n")
            
            print("正在计算第    ",i,"  个文档的tfidf")        #调试用  
            print("正在计算第    ",j,"  个单词")        #调试用
    print("以下为词典")
    f.write("以下为词典")
    for dic in tfidfDicLst:
        for key in dic:
          if dicall.__contains__(key) : 
              pass
          else :
            dicall[key]=1
            f.write(key)
            f.write("\n")
            print(key)
    f.close()  
    return tfidfDicLst

def Loadtxt2LstDic(dataSourcePath):
    n=0 #文档名在列表中的序号
    tfDicLst=[]
    docNameDic={}#存放文档名及其在列表中的序号，便于单独打印某个文档的tfidf       
    foldersLst=os.listdir(dataSourcePath)                                       #所有topic文件夹的列表
#     print("start")
    for topicFolder in foldersLst:  
        filesLst = os.listdir(dataSourcePath+topicFolder)
        for fileName in filesLst:                          
            filepath = dataSourcePath+topicFolder+'\\'+fileName            
            txt = open(filepath,'rb').read().decode('GBK', 'ignore')
            docNameDic[fileName]=n
            n+=1             
            texLst=Txt2wordLst(txt)                                                #对文本进行预处理：大小写、非英语内容、分词、词型转换、字母、去停            
            tfDictionary=txt2Dic(texLst)
            tfDicLst.append(tfDictionary)             
            print(n)        #调试用                    
    tf_idfDicLst=Tf_idfDicLst(tfDicLst)
    return docNameDic,tf_idfDicLst

def Vsm():                                               #计算TF-IDF并保存到矩阵中
    docNameDic,tf_idfDicLst=Loadtxt2LstDic(dataSourcePath)
#     f=open("vsmResult.txt","a")
#     for dic,docName in tf_idfDicLst,docNameDic:
#         f.write("\n")
#         f.write("\n")
#         f.write("以下是文档")
#         f.write(str(docName))
#         f.write("的Tf_Idf值")
#         f.write("\n")
#         for key in dic:
#             f.write(str(key))
#             f.write(':')
#             f.write(str(dic[key]))
#             f.write("\n")
#             print(key+':'+dic[key])
#     f.close()
    return 

if __name__=="__main__":
    dataSourcePath=os.getcwd()+"\\20news-18828\\"                               #数据集的存放路径
    Vsm()


