# coding:utf-8
# author: liubing
from VSM import *
import numpy

def Txt2Vex(txtDic,dsLst,dic):
    docCnt=0
    vexLst=[]
    tmp=[]
    for tpLst in dsLst:
        for wdDic in tpLst:
            docCnt+=1
            lst=list(wdDic)
            tmp.extend(lst)
    tmp.extend(list(txtDic))
    docCnt+=1
    dfDic=dict(Counter(tmp))    
    for wd in dic:
        if txtDic.__contains__(wd):
            tf=txtDic[wd]
            df=dfDic[wd]
            tfidf=cal_tfidf(tf,df,docCnt)
        else :
            tfidf=0                                                
        vexLst.append(tfidf)
    return vexLst

def Classify(vex1,dsLst,k):#计算给定vex1向量属于哪一类，返回类序号。（类序号从1-20）
    clsCnt=0
    topDisLst=[]
    dsDisLst=[]
    distance=[]
    minFrmLst=[]
    for topLst in dsLst:
        for wdLst in topLst:            
            dis=numpy.linalg.norm(vex1 - wdLst)#向量距离
            topDisLst.append(dis)
            distance.extend(dis)
        tmp=topDisLst        
        dsDisLst.append(tmp)
        topDisLst.clear()
    dis=sorted(distance)       
    for topLst in dsDisLst:
        clsCnt+=1
        for wdLst in topLst:
            for wd in wdLst:
                for i in k:
                    if wd == dis[i]:
                        minFrmLst[clsCnt]+=1 
    classNum=minFrmLst.index(max(minFrmLst))
    print("该文档分类为",classNum) 
    return classNum
 
def TestTxt(tfIdfLst,dic,dsLst,dataSourcePath):#对20%的测试文档中每一个进行分类，并计算准确率，函数返回准确率
    k=10    #KNN的k的取值。
    rightCnt=0
    wrongCnt=0
    cls=0#该文档实际是第x类
    foldersLst=os.listdir(dataSourcePath)
    for topicFolder in foldersLst:
        cls+=1 
        filesLst = os.listdir(dataSourcePath+topicFolder)
        path=dataSourcePath+topicFolder
        totalNum=len([name for name in os.listdir(path)])
        n=0
        for fileName in filesLst:
            if n<totalNum*0.8:
                n+=1
            else:#只对其余20%文档进行测试
                filepath = dataSourcePath+topicFolder+'\\'+fileName 
                txt = open(filepath,'rb').read().decode('GBK', 'ignore')                
                txt=Txt2wordLst(txt)          #对文本进行预处理：大小写、非英语内容、分词、词型转换、字母、去停            
                txtDic=txtReduce(txt)
                txtVexLst=Txt2Vex(txtDic,dsLst,dic)
                txtcls=Classify(txtVexLst,dsLst,k)          #类编号从1开始                
                if txtcls==cls:
                    rightCnt+=1
                else:
                    wrongCnt+=1 
                print("开始测试文档")
    rate=rightCnt/(rightCnt+wrongCnt)  
    print(rate)
    return rate
   
if __name__=="__main__":
    dataSourcePath=os.getcwd()+'\\'+"test.txt"       #数据集的存放路径
    t1=time.time()
    tfIdfLst,dic,dsLst=DataSource2TfIdfLst(dataSourcePath)
    rate=TestTxt(tfIdfLst,dic,dsLst,dataSourcePath)
    print(rate)#打印出k条件下的准确率。
    t2=time.time()
    print("总耗时为",t2-t1)