import numpy as np
from connected_graphs import Graph
from scipy.spatial.distance import cdist
from mincut import Graph as GraphMincut
import seaborn as sns
from matplotlib import  pyplot as plt
from Karger_mincut import  Graph as Karger_graph
from Karger_mincut import  fast_min_cut
import networkx as nx
import copy

class FINCH():
    def __init__(self, keys, faceClusters, speechClusters):
        self.keys = keys
        uniqueLabels = list(set(faceClusters))
        self.clusters = [[key for key, label_ in zip(keys, faceClusters) if label_ == label] \
            for label in uniqueLabels]
        self.speechClustersKeyWise = {key:label for key, label in zip(keys, speechClusters)}
        self.speechClustersLabelWise = {label: [key for key, label_ in zip(keys, speechClusters) \
            if label_ == label] for label in list(set(speechClusters))}
        self.th = 0.01

    def computeN1Matrix(self):
        clusterSpeechRep = [[self.speechClustersKeyWise[key] for key in cluster] \
            for cluster in self.clusters]
        similarityMatrix = np.zeros((len(self.clusters), len(self.clusters)))
        for i, clusteri in enumerate(self.clusters):
            for j, clusterj in enumerate(self.clusters):
                if j>i:
                    prob = 0.0
                    for speechLabel in self.speechClustersLabelWise.keys():
                        Pi = len([sLab for sLab in clusterSpeechRep[i] if sLab == speechLabel])\
                            / len(clusterSpeechRep[i])
                        Pj = len([sLab for sLab in clusterSpeechRep[j] if sLab == speechLabel])\
                            / len(clusterSpeechRep[j])
                        PspeechLabel = len(self.speechClustersLabelWise[speechLabel])/ len(self.keys)
                        prob += Pi*Pj*PspeechLabel
                    # print(prob)
                    similarityMatrix[i,j] = prob
                    similarityMatrix[j,i] = prob
             
            # return (similarityMatrix > self.th).astype(np.int)
            # plt.clf()
            # sns.heatmap(similarityMatrix)
            # plt.savefig('similarity_check.png', pi=300)
            # # th = 0.2
            N1 = [np.argmax(similarityMatrix[i]) for i in range(len(self.clusters))]
            # for i, n1 in N1:
            #     if similarityMatrix[i][n1] == 0:
            #         N1[i] = 0
            # print(N1)
            N1Matrix = np.zeros((len(self.clusters), len(self.clusters)))
            for idxi, n1i in enumerate(N1):
                for  idxj, n1j in enumerate(N1): 
                    
                    if idxi == idxj:
                        continue
                    N1Matrix[idxi,idxj] = (n1i == idxj) or (n1j == idxi) or (n1i == n1j)
            for i, n1 in enumerate(N1):
                if similarityMatrix[i][n1] <  0.01:
                    N1Matrix[i] = np.array([0]*len(N1))
                    N1Matrix[:,i] = np.array([0]*len(N1))
            return N1Matrix
        

    def N1Merge(self):
        N1Matrix = self.computeN1Matrix()
        # self.th *= 2
        # print(N1Matrix.tolist())
        connectedComponenets = Graph(N1Matrix.tolist()).connectedComponents()
        print(connectedComponenets)
        clusters = []
        for component in connectedComponenets:
            cluster  = []
            for idx in component:
                cluster.extend(self.clusters[idx])
            clusters.append(cluster)  
        self.clusters = clusters           
        # print(self.clusters)

    def convertTo(self):
        pred = {}
        for i, cluster in enumerate(self.clusters):
            for instance in cluster:
                pred[instance] = i
        return [pred[key] for key in self.keys]

    def cluster(self):
        clusters = []
        len_clusters= 0
        while (len(self.clusters)) > 5  and (len(self.clusters) != len_clusters):
            len_clusters= len(self.clusters)
            print('before merge', len(self.clusters))
            self.N1Merge()
            clusters.append(self.convertTo())
        return clusters



        