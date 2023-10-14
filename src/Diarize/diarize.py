import numpy as np
from sklearn.cluster import DBSCAN, AffinityPropagation
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import random, os
from collections import defaultdict
random.seed(1010)


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
    

    def clusterFaces(self):
        # use all the faces
        keys = list(self.faceFeatures.keys())
        distanceMatrix = self.pipe.distances.computeDistanceMatrix(keys, modality='face_raw')
        plotsDir = os.path.join(self.cacheDir, 'plots')
        os.makedirs(plotsDir, exist_ok=True)
        plt.clf()
        plt.imshow(distanceMatrix)
        plt.colorbar()
        plt.savefig(os.path.join(plotsDir, 'all_faces.png'), dpi=300)
        for i, keyi in enumerate(keys):
            facei = keyi
            for j, keyj in enumerate(keys):
                facej = keyj
                if facei == facej:
                    continue
                if self.faceTrackOverlap(facej, facei) > 0:

                    distanceMatrix[i,j] = 1.0
        plt.clf()
        plt.imshow(distanceMatrix)
        plt.colorbar()
        plt.savefig(os.path.join(plotsDir, 'all_faces_temporal_overlap_fixed.png'), dpi=300)
        
        # TODO: use the transformation of the speech distance matrix as a multiplier to the FD
        
        faceClusterer = AffinityPropagation(damping=0.5).fit(distanceMatrix)
        print(faceClusterer.labels_)
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

    def run(self):
        # self.clusterASD()
        self.clusterFaces()
        # return a dictionary of faceIds and labelIds

        return self.faceClusters