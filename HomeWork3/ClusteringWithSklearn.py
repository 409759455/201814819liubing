# coding:utf-8
# author:liubing

import time
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import SnowballStemmer
import nltk
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn import metrics
from sklearn.mixture import GaussianMixture


def TextLemmatization(corpus):                          # 词形还原
    wrds = []                                           # 还原后的list
    newCorpus = []
    newSentn = ""
    stemmer = SnowballStemmer("english")                # 词形还原
    for sentence in corpus:
        for word in nltk.word_tokenize(sentence):       # 每个文本打散成词
            wrds.append(stemmer.stem(word))
        newSentn = " ".join(wrds)
        newCorpus.append(newSentn)
        wrds.clear()
    return newCorpus


def Tfidf(corpus):
    vectorizer = CountVectorizer(stop_words='english')  # 去停用词、去字母小于2的词、平滑、正则化。
    csr_mat = vectorizer.fit_transform(corpus)          # 计算出词频稀疏矩阵
    transformer = TfidfTransformer()
    tfidf_csr_mat = transformer.fit_transform(csr_mat)
    print("字典长度", len(vectorizer.get_feature_names()))
    return tfidf_csr_mat


def loadDataSet(dataFile):
    corpus = []
    clsNoLst = []
    json_data = open(dataFile, 'rb').read().decode('GBK', 'ignore')
    jsonLst = json_data.split("\n")
    del jsonLst[-1]
    for jsn in jsonLst:
        jl = json.loads(jsn)
        corpus.append(jl["text"])
        clsNoLst.append(jl["cluster"])
    return corpus, clsNoLst


def print_NMI_Result(xCluster, clsNoLst):
    print("NMI: %s" % (metrics.normalized_mutual_info_score(xCluster, clsNoLst)))


def Kmeans(tfidf_csr_mat, clsNoLst):
    t1 = time.time()
    n_cluster_number = len(set(clsNoLst))
    kmCluster = KMeans(n_clusters=n_cluster_number, random_state=10).fit(tfidf_csr_mat)
    print_NMI_Result(kmCluster.labels_, clsNoLst)
    print("Kmeans耗时：", time.time() - t1)


def affinityPropagation(tfidf_csr_mat, clsNoLst):
    t1 = time.time()
    clustering = AffinityPropagation().fit(tfidf_csr_mat.toarray())
    print_NMI_Result(clustering.labels_, clsNoLst)
    print("affinityPropagation耗时：", time.time() - t1)


def meanShift(tfidf_csr_mat, clsNoLst):
    t1 = time.time()
    clustering = MeanShift().fit(tfidf_csr_mat.toarray()).predict(tfidf_csr_mat.toarray())
    print_NMI_Result(clustering, clsNoLst)
    print("meanShift耗时：", time.time() - t1)


def WardHierarchicalClustering(tfidf_csr_mat, clsNoLst):
    t1 = time.time()
    n_cluster_number = len(set(clsNoLst))
    clustering = AgglomerativeClustering(n_clusters=n_cluster_number, linkage='ward').fit(tfidf_csr_mat.toarray())
    print_NMI_Result(clustering.labels_, clsNoLst)
    print("WardHierarchicalClustering耗时：", time.time() - t1)


def spectralClustering(tfidf_csr_mat, clsNoLst):
    t1 = time.time()
    n_cluster_number = len(set(clsNoLst))
    DataCluster = SpectralClustering(n_clusters=n_cluster_number).fit(tfidf_csr_mat.toarray())
    print_NMI_Result(DataCluster.labels_, clsNoLst)
    print("spectralClustering耗时：", time.time() - t1)


def agglomerativeClustering(tfidf_csr_mat, clsNoLst):
    t1 = time.time()
    n_cluster_number = len(set(clsNoLst))
    DataCluster = AgglomerativeClustering(n_clusters=n_cluster_number).fit(tfidf_csr_mat.toarray())
    print_NMI_Result(DataCluster.labels_, clsNoLst)
    print("agglomerativeClustering耗时：", time.time() - t1)


def dBSCAN(tfidf_csr_mat, clsNoLst):
    t1 = time.time()
    clustering = DBSCAN(eps=1.13).fit(tfidf_csr_mat.toarray())
    print_NMI_Result(clustering.labels_, clsNoLst)
    print("dBSCAN耗时：", time.time() - t1)


def gaussianMixture(tfidf_csr_mat, clsNoLst):
    t1 = time.time()
    n_cluster_number = len(set(clsNoLst))
    GM = GaussianMixture(n_components=n_cluster_number, covariance_type='diag').fit(tfidf_csr_mat.toarray())
    clusters = GM.predict(tfidf_csr_mat.toarray())
    print_NMI_Result(clusters, clsNoLst)
    print("gaussianMixture耗时：", time.time() - t1)


def Test():
    corpus, clsNoLst = loadDataSet("Tweets.txt")
    corpus1 = TextLemmatization(corpus)
    tfidf_csr_mat = Tfidf(corpus1)
    # print(tfidf_csr_mat)
    # print(clsNoLst)

    print("***********Kmeans************")
    Kmeans(tfidf_csr_mat, clsNoLst)

    print("***********affinityPropagation************")
    affinityPropagation(tfidf_csr_mat, clsNoLst)

    print("***********meanShift************")
    meanShift(tfidf_csr_mat, clsNoLst)

    print("***********WardHierarchicalClustering************")
    WardHierarchicalClustering(tfidf_csr_mat, clsNoLst)

    print("***********spectralClustering************")
    spectralClustering(tfidf_csr_mat, clsNoLst)

    print("***********agglomerativeClustering************")
    agglomerativeClustering(tfidf_csr_mat, clsNoLst)

    print("***********dBSCAN************")
    dBSCAN(tfidf_csr_mat, clsNoLst)

    print("***********gaussianMixture************")
    gaussianMixture(tfidf_csr_mat, clsNoLst)


if __name__ == '__main__':
    t1 = time.time()
    print("开始...")
    Test()
    t2 = time.time()
    print("总耗时：", t2 - t1, "秒")
    print("结束...")
