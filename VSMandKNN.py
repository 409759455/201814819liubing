
# from textblob import TextBlob
# wiki = TextBlob("Python is a high-level, general-purpose programming language.")
#coding=UTF-8 
import os
dataSource="\\20news-18828\\"
a=0
topicFolders=os.listdir(os.getcwd()+dataSource)
for topicFolder in topicFolders:  
    path=os.getcwd()+dataSource+topicFolder
    fileNames = os.listdir(path)
    for filename in fileNames:                          
        filepath = path+'/'+filename
        for line in open(filepath,'rb'):                     
#             f.writelines(line)
#             f.write('\n')
            print(line)
            a=a+1
            print(a)
            print(path)
            print(filename)
            break