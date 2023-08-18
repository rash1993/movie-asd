'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-08 14:17:35
 * @modify date 2023-02-08 14:17:35
 * @desc [description]
 */'''

import os, sys
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import numpy as np
from local_utils import writeToPickleFile
import pickle as pkl 
from tqdm import tqdm

class Distances():
    def __init__(self, faceFeatures, speechFeatures, cacheDir, verbose):
        self.cacheDir = cacheDir
        self.faceFeatures = faceFeatures
        self.speechFeatures = speechFeatures
        self.verbose = verbose
        self.computeFaceTrackDistancesAll()

    def computeFaceTrackDistancesAll(self):
        # sourcery skip: do-not-use-bare-except
        fileName = os.path.join(self.cacheDir, 'faceDistancesAll.pkl')

        # check the cache
        if os.path.isfile(fileName):
            if self.verbose:
                print('reading face distances from cache')
            self.faceDistances =pkl.load(open(fileName, 'rb'))
            return

        # compute pairwise distances for all the face tracks
        if self.verbose:
            print(f'computing face distances and saving at: {fileName}')
        self.faceDistances = {}
        keys = list(self.faceFeatures.keys())
        for i in tqdm(range(len(keys)), desc='computing face distances'):
            disti = {
                keys[j]: cdist(
                    self.faceFeatures[keys[i]].reshape(1, -1),
                    self.faceFeatures[keys[j]].reshape(1, -1),
                    metric='cosine',
                )[0, 0]
                for j in range(i, len(keys))
            }
            self.faceDistances[keys[i]] = disti

        for keyi in keys:
            for keyj in keys:
                try:
                    self.faceDistances[keyi][keyj] = self.faceDistances[keyj][keyi]
                except:
                    self.faceDistances[keyj][keyi] = self.faceDistances[keyi][keyj]
        writeToPickleFile(self.faceDistances, fileName)

    def computeDistanceMatrix(self, keys, asd=None, modality='face'):
        distances = np.empty((len(keys), len(keys)))
        for i, keyi  in enumerate(keys):
            for j, keyj in enumerate(keys):
                if modality == 'speech':
                    distances[i,j] = cdist(self.speechFeatures[keyi].reshape(1, -1), \
                        self.speechFeatures[keyj].reshape(1, -1), metric='cosine')[0,0]
                elif modality == 'face':
                    face_i = asd[keyi]
                    face_j = asd[keyj]
                    distances[i,j] = self.faceDistances[face_i][face_j]
                else:
                    print('!!!!!!!!!! MODALITY NOT DEFINED !!!!!!!!!!')
        return distances

class Similarity():
    def __init__(self, measure):
        if measure == 'correlation':
            self.measure = 'correlation'
        else:
            sys.exit(f'similarity measure {measure} not implemented')

    def computeAvgSimilarity(self, audioDistances, faceDistances, avg=True):
        if self.measure == 'correlation':
            pr = [
                pearsonr(audioDistances[i], faceDistances[i])[0]
                for i in range(len(audioDistances))
            ]
            return np.mean(pr) if avg else pr