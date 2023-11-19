import numpy as np
import pickle as pkl 
import cv2, subprocess
from evaluate import bbIntersectionOverUnion 
from collections import Counter
from tools import writeToPklFile, readVideoFrames
from tqdm import tqdm
import os
from utils_cams import make_video
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.metrics import homogeneity_completeness_v_measure, mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate 
from c1c_clustering import FaceClustering
from finch import  FINCH
import simpleder
import random
random.seed(1010)

class Diarize():
    def __init__(self, asdFramework, speechFeatures, faceFeatures, cacheDir):
        self.pipe =  asdFramework
        self.speechFeatures = speechFeatures
        self.faceFeatures = faceFeatures
        self.cacheDir = cacheDir

    def clusterASD(self):
        keys = self.pipe.asd.keys()
        faceDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(keys,\
                                                                         self.pipe.asd, \
                                                                         modality='face')
        speechDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(keys, \
                                                                           modality='speech')
        facePred = self.speechfaceN1Merge(keys)
        clusters = {label: [key for key, label_ in zip(keys, facePred) if label_ == label] \
            for label in list(set(facePred))}
       
        gt = [self.speechIds[key] for key in keys]
        print(homogeneity_completeness_v_measure(gt, facePred), len(gt), len([d for d in gt if d == 'NA']))
        
        self.asdClusters = clusters
        
    def assignFacesToClusters(self, pred, keys):
        noiseKeys = [key for key, pred in zip(keys, pred) if pred<0]
        clusters  = {label: [key for key, pred_ in zip(keys, pred) if pred_ == label] \
            for label in list(set(pred)) if label >=0}
        clustersRep = {label: np.mean([self.pipe.face_tracks_embeddings[key] for key in clusters[label]], axis=0) for label in clusters.keys()}
        for key_ in noiseKeys:
            label_ = min(list(clusters.keys()), key=lambda label: \
                self.dist(self.pipe.face_tracks_embeddings[key_], clustersRep[label]))
            clusters[label_].append(key_)
        predOut = {}
        for label in clusters.keys():
            for key in clusters[label]:
                predOut[key] = label
        
        predOut = [predOut[key] for key in keys]
        return predOut
            
    def faceClustering(self, keys=None, speechKeys=None):
        faceTrackDistancesAll = self.pipe.idDistances.face_track_distances
        if keys == None:
            keys = faceTrackDistancesAll.keys()
            
            # keys = [value for key, value in self.pipe.asd.items()]
        faceDistanceMatrix = np.empty((len(keys), len(keys)))
        for i, keyi in enumerate(keys):
            for j, keyj in enumerate(keys):
                faceDistanceMatrix[i,j] = faceTrackDistancesAll[keyi][keyj]
                
        faceClusterer = DBSCAN(eps=0.1, min_samples=3, metric='precomputed').fit(faceDistanceMatrix)
        pred = self.assignFacesToClusters(faceClusterer.labels_, keys)
        self.evaluateFaceClusters(pred, keys, name='DBSCAN')

        # compute must links form speech
        if speechKeys != None:
            speechDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(speechKeys, modality='speech')
            faceDistanceMatrix = 0.8*faceDistanceMatrix + 0.2*speechDistanceMatrix
            # delta = 1.0
            # affinityMatrix = np.exp(- speechDistanceMatrix** 2 / (2. * delta ** 2))
            # th = np.percentile(affinityMatrix, 90)
            # mustlink = affinityMatrix > th

        # altering the temporally oveerlapping face tracks
        for i, keyi in enumerate(keys):
            st = self.pipe.face_tracks[keyi][0][0]
            et = self.pipe.face_tracks[keyi][-1][0]
            for j, keyj in enumerate(keys):
                st_ = self.pipe.face_tracks[keyj][0][0]
                et_ = self.pipe.face_tracks[keyj][-1][0]
                overlap = min(et, et_) - max(st, st_)
                if overlap >0.1:
                    faceDistanceMatrix[i,j] = 1.0
                # if mustlink[i,j]:
                #     faceDistanceMatrix[i,j] = 0.0
        
        faceClusterer = DBSCAN(eps=0.2, min_samples=3, metric='precomputed').fit(faceDistanceMatrix)
        if speechKeys:
            speechGt = [self.speechIds[key] for key in speechKeys]
            print('speech DBSCAN', homogeneity_completeness_v_measure(speechGt, faceClusterer.labels_))
        pred = self.assignFacesToClusters(faceClusterer.labels_, keys)
        self.evaluateFaceClusters(pred, keys, name='DBSCAN_cantlink')

        self.nameAllFaces(pred, keys)
        return pred
    
    def speechfaceN1Merge(self, speechKeys=None):
        if speechKeys == None:
            speechKeys = list(self.pipe.asd.keys())
        faceKeys = [self.pipe.asd[key] for key in speechKeys]

        faceDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(speechKeys, \
            asd=self.pipe.asd, modality='face')
        faceClusterer = DBSCAN(eps=0.15, min_samples=3, metric='precomputed').fit(faceDistanceMatrix)
        predFace = self.assignFacesToClusters(faceClusterer.labels_, faceKeys)
        self.evaluateFaceClusters(predFace, faceKeys, name='face')

        for i, keyi in enumerate(faceKeys):
            st = self.pipe.face_tracks[keyi][0][0]
            et = self.pipe.face_tracks[keyi][-1][0]
            for j, keyj in enumerate(faceKeys):
                st_ = self.pipe.face_tracks[keyj][0][0]
                et_ = self.pipe.face_tracks[keyj][-1][0]
                overlap = min(et, et_) - max(st, st_)
                if overlap >0.1:
                    faceDistanceMatrix[i,j] = 1.0

        faceClusterer = DBSCAN(eps=0.15, min_samples=3, metric='precomputed').fit(faceDistanceMatrix)
        predFace = self.assignFacesToClusters(faceClusterer.labels_, faceKeys)
        self.evaluateFaceClusters(predFace, faceKeys, name='face')

        speechDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(speechKeys, modality='speech')
        delta = 1.0
        speechAffiittyMatrix = np.exp(- speechDistanceMatrix ** 2 / (2. * delta ** 2))
        numClusters = 15
        speechClusterer = SpectralClustering(n_clusters=numClusters, random_state=0,\
            assign_labels='kmeans', affinity='precomputed').fit(speechAffiittyMatrix)
        self.evaluateFaceClusters(speechClusterer.labels_, faceKeys, 'speech')

        finch = FINCH(faceKeys, predFace, speechClusterer.labels_ )
        clusters = finch.cluster()
        for cluster in clusters:
            self.evaluateFaceClusters(cluster, faceKeys, name=f'N1Merge_{len(cluster)}')
        
        self.nameAllFaces(clusters[0], faceKeys)
        return clusters[0]

    def speechClustering(self, keys=None):
        '''
        cluster the speech segments for several values of K
        '''
        if keys == None:
            keys = set([key for key, id_ in self.speechIds.items() if id_ != 'NA']).intersection(set(list(self.pipe.speech_feats.keys())))
        distance_matrix = self.pipe.idDistances.computeDistanceMatrix(keys, modality='speech')
        delta = 1.0
        affinity_matrix = np.exp(- distance_matrix ** 2 / (2. * delta ** 2))

        numChars = len(set([charID for charID in self.speechIds.values() if charID != 'NA']))
        print(numChars)

        # for numClusters in range(5, 14):
        numClusters = numChars
        speechClusterer = SpectralClustering(n_clusters=numClusters, random_state=0,\
            assign_labels='kmeans', affinity='precomputed').fit(affinity_matrix)
        pred = speechClusterer.labels_
        gt = [self.speechIds[key] for key in keys]
        unique_gt = list(set(gt))

        self.asdClusters = {label: [key for key, pred_ in zip(keys, pred) if pred_ == label]\
            for label in list(set(pred))}
        print(self.computeSimpleDer(keys, pred))
        
        # print('speech clustering', numClusters, homogeneity_completeness_v_measure(gt, pred), len(set(pred)))
        # keys = self.pipe.asd.keys()
        faceDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(keys,self.pipe.asd, modality='face')
        speechDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(keys, modality='speech')
        corr = self.pipe.similarity.computeAvgSimilarity(speechDistanceMatrix, faceDistanceMatrix, avg=False)
        keyCorr = {key:corr_ for key, corr_ in zip(keys, corr)}

        pred = [label for label, key in zip(speechClusterer.labels_, keys) if keyCorr[key]>0.1]
        selKeys = [key for key in keys if keyCorr[key]>0.1]
        print('speech clustering for selected keys', homogeneity_completeness_v_measure([self.speechIds[key] for key in selKeys], \
            pred))
        
        clusters = {label: [key for key, label_  in zip(selKeys, pred) if label_ == label] for label in list(set(pred))}
        self.asdClusters = clusters
        return self.computeSimpleDer(keys, pred)
        # self.evaluateFaceClusters(pred, [self.pipe.asd[key] for key in  selKeys], name='speech clustering')
    
    def assignRemainingSpeech(self):
        assigned = []
        for label, cluster in self.asdClusters.items():
            assigned.extend(cluster)

        keys = [key for key in self.pipe.voice_face_tracks.keys() if key not in assigned]
        keys = set(keys).intersection(set(self.pipe.speech_feats.keys()))

        assignments = []
        for key in keys:
            distances = []
            for label, cluster in self.asdClusters.items():
                distance = 0.0
                for keyB in cluster:
                    distance += cdist(self.pipe.speech_feats[key].reshape(1,-1),\
                        self.pipe.speech_feats[keyB].reshape(1, -1), metric='cosine')[0,0]
                distances.append([label, distance/len(cluster)])
            distances.sort(key=lambda x: x[1])
            assignments.append([key, distances[0][0]])

        for assignment in assignments:
            self.asdClusters[assignment[1]].append(assignment[0])   
        
        predKeys = []
        predLabels = []
        for label, cluster in self.asdClusters.items():
            for key in cluster:
                predLabels.append(label)
                predKeys.append(key)
        gt = [self.speechIds[key] for key in predKeys]
        print(self.computeSimpleDer(predKeys, predLabels))
        
        print(f'assigned: {len(assigned)}, toBeAssigned: {len(assignments)}')
        print(homogeneity_completeness_v_measure(gt, predLabels), len([d for d in gt if d == 'NA']))
        return self.computeSimpleDer(predKeys, predLabels)
    
    def clusterASDFacesHAC(self):
        # keys = self.pipe.asd.keys()
        keys = set([key for key, id_ in self.speechIds.items() if id_ != 'NA']).intersection(set(list(self.pipe.speech_feats.keys())))
        faceDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(keys,self.pipe.asd, modality='face')
        speechDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(keys, modality='speech')
        corr = self.pipe.similarity.computeAvgSimilarity(speechDistanceMatrix, faceDistanceMatrix, avg=False)
        keyCorr = {key:corr_ for key, corr_ in zip(keys, corr)}
        # selecting high confidence ASD
        keys = [key for key in keys if keyCorr[key]> 0.1]

        speechDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(keys, modality='speech')
        faceDistanceMatrix = self.pipe.idDistances.computeDistanceMatrix(keys, asd=self.pipe.asd, modality='face')
        distanceMatrix = faceDistanceMatrix*0.8 + speechDistanceMatrix*0.2
        # for i, keyi in enumerate(keys):
        #     facei = self.pipe.asd[keyi]
        #     st = self.pipe.face_tracks[facei][0][0]
        #     et = self.pipe.face_tracks[facei][-1][0]
        #     for j, keyj in enumerate(keys):
        #         facej = self.pipe.asd[keyj]
        #         if self.faceTrackOverlap(facej, st, et):
        #             distanceMatrix[i,j] = 1.0
        
        numChars = len(set([charID for charID in self.speechIds.values() if charID != 'NA']))
        print(numChars)

        numClusters = numChars
        numClusterObtained = 0

        while numClusterObtained != numChars: 
            clusterer = AgglomerativeClustering(n_clusters=numClusters, affinity='precomputed', \
                linkage='average').fit(distanceMatrix)
            facekeys = [self.pipe.asd[key] for key in keys]
            numClusterObtained, vscore = self.evaluateFaceClusters(clusterer.labels_, facekeys, name='face')
            numClusters += 1

        facePred = clusterer.labels_
        clusters = {label: [key for key, label_ in zip(keys, facePred) if label_ == label] \
            for label in list(set(facePred))}

        speechPred = [self.speechIds[key] for key in keys]
        print('speech label from faces', homogeneity_completeness_v_measure(speechPred, facePred))
        self.asdClusters = clusters

    def diarizeASDFaces(self):
        self.clusterASD()
        # self.clusterASDFacesHAC()
        # self.nameClustersUsingGT()
        faceDER = self.assignRemainingSpeech()
        # return faceDER
        self.faceClustering()
        # self.computeDER()
    
        self.visualize()

    def run(self):
        self.getGTFaceIDs()
        # self.faceClustering()
        self.getGTSpeechIDs()
        # self.speechfaceN1Merge()
        # speechDER = self.speechClustering()
        # assigneSpeechDER = self.assignRemainingSpeech()
        # print('assigned speech', assigneSpeechDER)
        # self.findingKExpt()
        # self.visualize()
        # self.AudioAssistedFaceClustering()
        self.diarizeASDFaces()
        # print(faceDER)
        # f = open('diarization.txt', 'a')
        # print(f'{self.pipe.video_name},{speechDER},{faceDER}', file=f)
        # f.close() 