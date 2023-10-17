import numpy as np
from sklearn.cluster import DBSCAN, AffinityPropagation
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import random, os
from collections import defaultdict
random.seed(1010)

class Graph:
        # init function to declare class variables
        def __init__(self, adj):
                matrixFlag = True
                N = len(adj)
                for i in adj:
                        if len(i) != N:
                                matrixFlag = False
                                break
                if matrixFlag:
                        self.adj = [[idx for idx, ele in enumerate(row) if ele >0 ] for row in adj]
                else:
                        self.adj = adj
                # print(self.adj)

        def DFSUtil(self, temp, v, visited):

                # Mark the current vertex as visited
                visited[v] = True

                # Store the vertex to list
                temp.append(v)

                # Repeat for all vertices adjacent
                # to this vertex v
                for i in self.adj[v]:
                        if visited[i] == False:

                                # Update the list
                                temp = self.DFSUtil(temp, i, visited)
                return temp

        # Method to retrieve connected components
        # in an undirected graph
        def connectedComponents(self):
                visited = []
                cc = []
                for i in range(len(self.adj)):
                        visited.append(False)
                for v in range(len(self.adj)):
                        if visited[v] == False:
                                temp = []
                                cc.append(self.DFSUtil(temp, v, visited))
                return cc

class Diarize():
    def __init__(self, asdFramework, cacheDir):
        self.pipe =  asdFramework
        self.speechFeatures = asdFramework.speechFeatures
        self.faceFeatures = asdFramework.faceFeatures
        self.cacheDir = cacheDir
    
    def dist(self, x, y):
        return cdist(x.reshape(1, -1), y.reshape(1, -1), metric='cosine')[0,0]

    def faceTrackOverlap(self, facei, facej):
        # function to return overlap between two face tracks. 
        sti = self.pipe.faceTracks[facei][0][0]
        eti = self.pipe.faceTracks[facei][-1][0]
        stj = self.pipe.faceTracks[facej][0][0]
        etj = self.pipe.faceTracks[facej][-1][0]
        overlap = min(etj, eti) - max(sti, stj)
        return overlap

    def assignFacesToClusters(self, pred, keys):
        # keys which are considered noise in DBSCAN
        noiseKeys = [key for key, pred in zip(keys, pred) if pred<0]
        clusters  = {label: [key for key, pred_ in zip(keys, pred) if pred_ == label] \
            for label in list(set(pred)) if label >=0}
        # cluster representations as mean of the isntance representations
        clustersRep = {label: np.mean([self.pipe.faceFeatures[self.pipe.asd[key]] \
                       for key in clusters[label]], axis=0) \
                       for label in clusters.keys()}
        # assign the noisy instnaces to the defined clusters
        for key_ in noiseKeys:
            # TODO: check that noisy face is not assigned to 
            # the cluster with temporally overlapping face tracks
            label_ = min(list(clusters.keys()), key=lambda label: \
                self.dist(self.pipe.faceFeatures[self.pipe.asd[key_]], clustersRep[label]))
            clusters[label_].append(key_)
        predOut = {}
        for label in clusters.keys():
            for key in clusters[label]:
                predOut[key] = label
        
        predOut = [predOut[key] for key in keys]
        return predOut
    
    def clusterASD(self):
        # keys = self.pipe.asd.keys()
        keys = set([key for key, id_ in self.pipe.asd.items() if id_ != 'NA']).\
                    intersection(set(list(self.speechFeatures.keys())))
        faceDistanceMatrix = self.pipe.distances.\
                             computeDistanceMatrix(keys,self.pipe.asd, \
                                                   modality='face')
        speechDistanceMatrix = self.pipe.distances.\
                               computeDistanceMatrix(keys, \
                                                     modality='speech')
        # Distance matrix as a linear combination of the face and speech distance matrices
        distanceMatrix = faceDistanceMatrix
        # distanceMatrix = np.multiply(faceDistanceMatrix, speechDistanceMatrix)

        # Changing the distnace between the face-tracks with non-zero temporal overlap to 1.0 
        for i, keyi in enumerate(keys):
            facei = self.pipe.asd[keyi]
            for j, keyj in enumerate(keys):
                facej = self.pipe.asd[keyj]
                if self.faceTrackOverlap(facej, facei) > 0:
                    distanceMatrix[i,j] = 1.0
        
        # Cluster ASD faces using DBSCAN
        # faceClusterer = DBSCAN(eps=0.1, min_samples=1,  metric='precomputed').\
        #                        fit(distanceMatrix)
        faceClusterer = AffinityPropagation(damping=0.9).fit(distanceMatrix)
        print(faceClusterer.labels_)
        cluster_Ids = self.assignFacesToClusters(faceClusterer.labels_, keys)
        self.asdClusters = {key: id for key, id in zip(keys, cluster_Ids)}
    
    def speechInfo(self):
        # asd keys
        keys = self.pipe.asd.keys()
        speechDistanceMatrix = self.pipe.distances.\
                               computeDistanceMatrix(keys, \
                                                     modality='speech')
        
        faceDistanceMultiplier = {}
        for i, keyi in enumerate(keys):
            faceKeyi = self.pipe.asd[keyi]
            faceDistanceMultiplier[faceKeyi] = {}
            for j, keyj in enumerate(keys):
                faceKeyj = self.pipe.asd[keyj]
                if faceKeyj in faceDistanceMultiplier[faceKeyi].keys():
                    faceDistanceMultiplier[faceKeyi][faceKeyj].append(speechDistanceMatrix[i, j])
                else:
                    faceDistanceMultiplier[faceKeyi][faceKeyj] = [speechDistanceMatrix[i, j]]
        th_same = 0.2
        th_notsame = 0.6
        for keyi in faceDistanceMultiplier.keys():
            for keyj in faceDistanceMultiplier[keyi].keys():
                multiplier = np.mean(faceDistanceMultiplier[keyi][keyj])
                if multiplier >= th_notsame:
                    multiplier = min(4, np.exp(10*(multiplier - th_notsame)))
                elif multiplier <= th_same:
                    multiplier = max(0.25, np.exp(10*(multiplier - th_same)))
                else:
                    multiplier = 1
                faceDistanceMultiplier[keyi][keyj] = multiplier

        # make sure the faceDistance multiplier is a square matrix
        for keyi in faceDistanceMultiplier.keys():
            for keyj in faceDistanceMultiplier[keyi].keys():
                try:
                    faceDistanceMultiplier[keyi][keyj] = faceDistanceMultiplier[keyj][keyi]
                except:
                    faceDistanceMultiplier[keyj][keyi] = faceDistanceMultiplier[keyi][keyj]
        return faceDistanceMultiplier

    def combineFaceClustersUsingSpeech(self, faceClusters):
        # faceIdToCluster = {value: key for key, value in faceClusters.items()}
        
        faceClusterSpeechRep = {key: [] for key in list(set(list(faceClusters.values())))}
        print(faceClusterSpeechRep.keys())
        for speechKey, faceKey in self.pipe.asd.items():
            faceClusterSpeechRep[faceClusters[faceKey]].append(self.pipe.speechFeatures[speechKey].numpy().reshape(-1))
        noSpeechKeys = [key for key, value in faceClusterSpeechRep.items() if len(value) == 0]
        faceClusterSpeechRep = {key: np.mean(np.array(value), axis=0) \
                                for key, value in faceClusterSpeechRep.items() if key not in noSpeechKeys}
        faceClusterSpeechDistanceMatrix = np.zeros((len(faceClusterSpeechRep.keys()), \
                                                   len(faceClusterSpeechRep.keys())))
        keys = list(faceClusterSpeechRep.keys())
        for i, keyi in enumerate(keys):
            for j, keyj in enumerate(keys):
                faceClusterSpeechDistanceMatrix[i,j] = cdist(faceClusterSpeechRep[keyi].reshape(1, -1),\
                                                             faceClusterSpeechRep[keyj].reshape(1, -1),\
                                                             metric='cosine')[0,0]
        
        th = 0.2
        adj = faceClusterSpeechDistanceMatrix < th
        adj = adj.astype(int)
        connected_components = Graph(adj).connectedComponents()
        clusterMap = {}
        for components in connected_components:
            # finding a new cluster id for the connected components
            for i in range(1000):
                if i in list(clusterMap.keys()) + noSpeechKeys:
                    continue
                else:
                    cluster_id = i
                    break
            for cluster in components:
                clusterMap[cluster] = cluster_id
        for cluster in noSpeechKeys:
            clusterMap[cluster] = cluster
        print(clusterMap)
        for key in faceClusters.keys():
            faceClusters[key] = clusterMap[faceClusters[key]]
        return faceClusters

    def clusterFaces(self):
        # use all the faces
        keys = list(self.faceFeatures.keys())
        distanceMatrix = self.pipe.distances.computeDistanceMatrix(keys, modality='face_raw')
        plotsDir = os.path.join(self.cacheDir, 'plots')
        os.makedirs(plotsDir, exist_ok=True)
        plt.clf()
        plt.imshow(distanceMatrix, vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(plotsDir, 'all_faces.png'), dpi=300)

        # incorporate speech info
        print(np.sum(distanceMatrix > 1.0))
        useSpeechInfo = False
        if useSpeechInfo:
            faceDistanceMultiplier = self.speechInfo()
            distanceMatrix = self.pipe.distances.computeDistanceMatrix(keys, modality='face_raw', return_dict=True)
            for keyi in faceDistanceMultiplier.keys():
                for keyj in faceDistanceMultiplier[keyi].keys():
                    distanceMatrix[keyi][keyj] *= faceDistanceMultiplier[keyi][keyj]
        
            # construct the distance matrix from dict format
            distanceMatrix_ = np.zeros((len(keys), len(keys)))
            for i, keyi in enumerate(keys):
                for j, keyj in enumerate(keys):
                    distanceMatrix_[i,j] = distanceMatrix[keyi][keyj]
            distanceMatrix = distanceMatrix_
            plt.clf()
            plt.imshow(distanceMatrix, vmin=0, vmax=2)
            plt.colorbar()
            plt.savefig(os.path.join(plotsDir, 'all_faces_speech_info.png'), dpi=300)

        # constraint: distance between the temporally overlapping facetracks is 1.0 (max od the distance matrix)
        for i, keyi in enumerate(keys):
            facei = keyi
            for j, keyj in enumerate(keys):
                facej = keyj
                if facei == facej:
                    continue
                if self.faceTrackOverlap(facej, facei) > 0:
                    distanceMatrix[i,j] = np.max(distanceMatrix)

        plt.clf()
        plt.imshow(distanceMatrix, vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(plotsDir, 'all_faces_temporal_overlap_fixed.png'), dpi=300)
        
        # distanceMatrix = 1 - distanceMatrix
        delta = 10
        distanceMatrix = np.exp(- distanceMatrix ** 2 / (2. *delta ** 2))
        faceClusterer = AffinityPropagation(damping=0.5).fit(distanceMatrix)
        self.faceClusters = {key:clusterId for key, clusterId in zip(keys, faceClusterer.labels_)}

        # arrange the keys according to clusters
        clusters = defaultdict(lambda: [])
        for key, id in self.faceClusters.items():
            clusters[id].append(key)
        arranged_keys = []
        for value in clusters.values():
            arranged_keys.extend(value)
        
        distanceMatrix = self.pipe.distances.computeDistanceMatrix(arranged_keys, modality='face_raw')
        plt.clf()
        plt.imshow(distanceMatrix)
        plt.colorbar()
        plt.savefig(os.path.join(plotsDir, 'all_faces_clustered_keys.png'), dpi=300)

        # combine the face clusters based on speech representations
        self.faceClusters = self.combineFaceClustersUsingSpeech(self.faceClusters)
        # arrange the keys according to clusters
        clusters = defaultdict(lambda: [])
        for key, id in self.faceClusters.items():
            clusters[id].append(key)
        arranged_keys = []
        for value in clusters.values():
            arranged_keys.extend(value)
        
        distanceMatrix = self.pipe.distances.computeDistanceMatrix(arranged_keys, modality='face_raw')
        plt.clf()
        plt.imshow(distanceMatrix)
        plt.colorbar()
        plt.savefig(os.path.join(plotsDir, 'all_faces_clustered_keys_combined.png'), dpi=300)

    def run(self):
        # self.clusterASD()
        self.clusterFaces()
        # return a dictionary of faceIds and labelIds

        return self.faceClusters