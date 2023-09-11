'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-08 19:42:51
 * @modify date 2023-02-08 19:42:51
 * @desc [description]
 */'''
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from random import shuffle
from scipy.stats import pearsonr
SEED = 4

class SpeechFaceAssociation():
    def __init__(self, 
                 cacheDir,
                 speechFaceTracks, 
                 marginalFaceTracks,
                 similarity, 
                 distances, 
                 faceTracks, 
                 guides,
                 verbose=False):
        self.cacheDir = cacheDir
        self.speechFaceTracks = speechFaceTracks
        self.marginalFaceTracks = marginalFaceTracks
        self.similarity = similarity
        self.distances = distances
        self.faceTracks = faceTracks
        self.guides = guides
        self.verbose = verbose

    def getFaceTrackArea(self, faceTrackId):
        faceTrack = self.faceTracks[faceTrackId]
        area = 0.0
        for box in faceTrack:
            area += (box[3]-box[1])*(box[4]-box[2])
        
        return area / len(faceTrack)
    
    def initializeASD(self, speechKeys):  # sourcery skip: do-not-use-bare-except
        asd = {} # assigned face track for each speech segment ['speechSegment': 'faceTrack']
        posGuides = [] # list of speech segments
        negGuides = {key_:[] for key_ in speechKeys} # dictionary of list of negative guides fro each speech segment
        if self.guides:
            guidesPredScores = self.guides['scores']
            posTh = self.guides['posTh']
            negTh = self.guides['negTh']
            for key_ in speechKeys:
                # selecting the face tracks with highest score from guides
                tracks = self.speechFaceTracks[key_]['face_tracks']
                if len(tracks) == 0:
                    # not considering keys where no face tracks overlap
                    continue
                guideScores_ = [[track[0], guidesPredScores[key_][track[0]]] \
                                    for track in tracks \
                                    if not np.isnan(float(guidesPredScores[key_][track[0]]))]
                guideScores_.sort(key=lambda x: x[1], reverse=True)
                asd[key_] = (
                    guideScores_[0][0]
                    if guideScores_
                    # initializing wiht largest face
                    else max(tracks, key=lambda x: self.getFaceTrackArea(x[0]))[0]
                    # initializing with random face
                    # else tracks[np.random.randint(0, len(tracks), size=1)[0]][0]
                )
                
                # determining if positive guide
                try:
                    if guidesPredScores[key_][asd[key_]] >= posTh:
                        posGuides.append(key_)
                except:
                    pass
                
                # determining the negative guides for the speech segment
                negGuides[key_] = [trackId for trackId, score in guideScores_ \
                                    if (score < negTh) and (not np.isnan(float(score )))] 
                
            removeKeys = []
            for key_ in asd.keys():
                    if asd[key_] in negGuides[key_]:
                        removeKeys.append(key_)
            asd = {k:v for k, v in asd.items() if k not in removeKeys}
        else:
            # randomly initializing the asd
            for key_ in speechKeys:
                tracks = self.speechFaceTracks[key_]['face_tracks']
                if len(tracks) == 0:
                    continue
                # initializing with max area face track
                asd[key_] = max(tracks, key=lambda x: self.getFaceTrackArea(x[0]))[0]
                # asd[key_] = tracks[np.random.randint(0, len(tracks), size=1)[0]][0]
        return asd, posGuides, negGuides

    def handler(self, partitionLen):
        speechKeys = list(self.speechFaceTracks.keys())
        # speechKeys.sort(key=lambda x: self.speechFaceTracks[x]['speech'][0]) # sorting with start time
        shuffle(speechKeys)
        ASD = {}
        if str(partitionLen).isdigit():
            numPartitions = int(np.ceil(len(speechKeys)/partitionLen))
            partitions = [
                speechKeys[i * partitionLen : (i + 1) * partitionLen]
                for i in range(numPartitions)
            ]
            if len(partitions[-1])<5:
                partitions[-2] = partitions[-2] + partitions[-1]
                del partitions[-1]
            for partitionNum, partition in enumerate(partitions):
                print(f'optimizing the partition {partitionNum} of {len(partitions)}')
                asd, posGuides, negGuides = self.initializeASD(partition)
                asd = self.findSpeechFaceAssociationPartion(asd, posGuides, negGuides)
                # asd = self.offscreenSpeakercorrection2(asd)
                ASD.update(asd)
        else:
            asd, posGuides, negGuides = self.initializeASD(speechKeys)
            asd = self.findSpeechFaceAssociationPartion(asd, posGuides, negGuides)
            asd = self.offscreenSpeakercorrection2(asd)
            ASD.update(asd)
        ASD =  self.offscreenSpeakercorrection(ASD)
        return ASD
    
    def offscreenSpeakercorrection(self, asd, th=0.1):
        speechKeys = list(asd.keys())
        audioDistances = self.distances.computeDistanceMatrix(\
                                    speechKeys, modality='speech')
        faceDistances = self.distances.computeDistanceMatrix(\
                                    speechKeys, asd=asd, modality='face')
        corr = self.similarity.computeAvgSimilarity(\
                                    audioDistances, faceDistances, avg=False)
        offScreenSpeechKeys = [key_ for key_, corr_ in zip(speechKeys, corr) if corr_ < th]
        asd = {key:asd[key] for key in asd.keys() if key not in offScreenSpeechKeys}
        return asd
    
    def getMarginalDistances(self, asd, currCorr):
        keys = list(asd.keys())
        audioDistances = self.distances.computeDistanceMatrix(\
                                        keys, modality='speech') 
        marginal_distances = []
        for i, key in tqdm(enumerate(keys), desc='computing marginal distances'):
            audioDistanceVector = audioDistances[i]
            currCorri = currCorr[i]
            mdistances = []
            for trackId  in self.marginalFaceTracks[key]['face_tracks']:
                if trackId[0] != asd[key]:
                    asd_ = asd.copy()
                    asd_[key] = trackId[0]
                    keys_ = keys
                    faceDistances = self.distances.computeDistanceMatrix(\
                                        keys_, asd=asd_, modality='face')
                    faceDistanceVector = faceDistances[i]
                    newCorri = pearsonr(audioDistanceVector, faceDistanceVector)[0]
                    mdistances.append([trackId[0], newCorri - currCorri])
            if len(mdistances):
                mdistances.sort(key=lambda x: x[1], reverse=True)
                marginal_distances.append([key] + mdistances[0])
        return marginal_distances

    def offscreenSpeakercorrection2(self, asd):
        # optimize the matrix to the maximum value by replacing the faces with the neighboring \
        # faces in the matrix. Starting with the highly likely off-screen speaker move to the least likely
        last_corr = 0.0
        keys = list(asd.keys())
        audioDistances = self.distances.computeDistanceMatrix(keys=keys, modality='speech')
        faceDistances = self.distances.computeDistanceMatrix(keys=keys, asd=asd, modality='face')
        last_corr = self.similarity.computeAvgSimilarity(audioDistances, faceDistances, avg=False)
        curr_corr = last_corr
        while True:
            last_corr = curr_corr.copy()
            marginal_distances = self.getMarginalDistances(asd, curr_corr)
            sel_key = max(marginal_distances, key=lambda x: x[2])
            if (sel_key[2] > 0) and (sel_key[2] > 0.1):
                asd[sel_key[0]] = sel_key[1]
                faceDistances = self.distances.computeDistanceMatrix(keys=keys, asd=asd, modality='face')
                curr_corr = self.similarity.computeAvgSimilarity(audioDistances, faceDistances, avg=False)
                print(f'curr_corr: {np.mean(curr_corr)} | sel_key: {sel_key}, last_corr: {np.mean(last_corr)}')
            else:
                break
            if (np.mean(curr_corr) - np.mean(last_corr))/np.mean(last_corr) < 0.01:
                break
        return asd
            
    def findSpeechFaceAssociationPartion(self, asd, posGuides, negGuides):
        # sourcery skip: low-code-quality
        if self.verbose:
            negGuidesFaceTracksCount = np.sum(len(faceTracks_) for faceTracks_ in negGuides.values())
            faceTracksCount = np.sum(len(self.speechFaceTracks[key_]['face_tracks']) for key_ in asd.keys())
            print(f'positive guides: {len(posGuides)}/ {len(asd.keys())}')
            print(f'negtive guided face tracks: {negGuidesFaceTracksCount}/{faceTracksCount}')
        # remove keys using the negative guides
        removeKeysCount = 0
        removeKeys = []
        for key_ in asd.keys():
            trackIds = [track[0] for track in self.speechFaceTracks[key_]['face_tracks']\
                        if track[0] not in negGuides[key_]]
            if not len(trackIds):
                removeKeys.append(key_)
                removeKeysCount += 1
        asd = {k:v for k, v in asd.items() if k not in removeKeys}
        
        speechKeys = list(asd.keys())
        lastCorr = 0.0
        maxEpoch = 20
        for epoch in range(maxEpoch):
            random.Random(SEED).shuffle(speechKeys)
            audioDistances = self.distances.computeDistanceMatrix(\
                                        speechKeys, modality='speech')
            faceDistances = self.distances.computeDistanceMatrix(\
                                        speechKeys, asd=asd, modality='face')
            currentCorr = self.similarity.computeAvgSimilarity(\
                                        audioDistances, faceDistances)
            if self.verbose:
                print(f'epoch: {epoch} | corr: {currentCorr}')
            for i, keyi in enumerate(tqdm(speechKeys, \
                desc=f'optimizing epoch {epoch}')):
                if keyi in posGuides:
                    continue
                faceTracks = self.speechFaceTracks[keyi]['face_tracks']
                corrTracker_ = [[asd[keyi], currentCorr, faceDistances]]
                for faceTrack in faceTracks:
                    if faceTrack[0] in negGuides[keyi]:
                        continue
                    fdRep = [self.distances.faceDistances[faceTrack[0]][asd[keyj]]\
                        for keyj in speechKeys]
                    fdRep[i] = 0.0
                    faceDistanceCopy = deepcopy(faceDistances)
                    faceDistanceCopy[i] = fdRep
                    faceDistanceCopy[:,i] = fdRep
                    corrTracker_.append([faceTrack[0], self.similarity.computeAvgSimilarity(\
                                            audioDistances, faceDistanceCopy), faceDistanceCopy])
                asdFaceTrack = max(corrTracker_, key=lambda x: x[1])
                asd[keyi] = asdFaceTrack[0]
                currentCorr = asdFaceTrack[1]
                faceDistances = asdFaceTrack[2]
            epochCorr = self.similarity.computeAvgSimilarity(audioDistances, faceDistances)
            diffCorr = (epochCorr - lastCorr)/lastCorr
            if diffCorr < 0.2:
                break
            else:
                lastCorr = epochCorr
        if self.verbose:
            print(f'epoch: last | corr: {epochCorr}')
        return asd
        