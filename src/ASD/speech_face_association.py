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
SEED = 4

class SpeechFaceAssociation():
    def __init__(self, 
                 cacheDir,
                 speechFaceTracks, 
                 similarity, 
                 distances, 
                 faceTracks, 
                 guides,
                 verbose=False):
        self.cacheDir = cacheDir
        self.speechFaceTracks = speechFaceTracks
        self.similarity = similarity
        self.distances = distances
        self.faceTracks = faceTracks
        self.guides = guides
        self.verbose = verbose

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
                guideScores_ = [[track[0], guidesPredScores[track[0]]] \
                                    for track in tracks if guidesPredScores[track[0]] != 'nan']
                guideScores_.sort(key=lambda x: x[1])
                asd[key_] = (
                    guideScores_[0][0]
                    if guideScores_
                    else tracks[np.random.randint(0, len(tracks), size=1)[0]][0]
                )
                
                # determining if positive guide
                try:
                    if guidesPredScores[asd[key_]] >= posTh:
                        posGuides.append(key_)
                except:
                    pass
                
                # determining the negative guides for the speech segment
                negGuides[key_] = [trackId for trackId, score in guideScores_ \
                                    if score < negTh] 
        else:
            # randomly initializing the asd
            for key_ in speechKeys:
                tracks = self.speechFaceTracks[key_]['face_tracks']
                if len(tracks) == 0:
                    continue
                asd[key_] = tracks[np.random.randint(0, len(tracks), size=1)[0]][0]
        return asd, posGuides, negGuides

    def handler(self, partitionLen):
        speechKeys = list(self.speechFaceTracks.keys())
        speechKeys.sort(key=lambda x: self.speechFaceTracks[x]['speech'][0]) # sorting with start time
        ASD = {}
        if str(partitionLen).isdigit():
            numPartitions = int(np.ceil(len(speechKeys)/partitionLen))
            partitions = [
                [speechKeys[i * partitionLen : (i + 1) * partitionLen]]
                for i in range(numPartitions)
            ]
            if len(partitions[-1])<5:
                partitions[-2] = partitions[-2] + partitions[-1]
                del partitions[-1]

            for partition in partitions:
                asd, posGuides, negGuides = self.initializeASD(partition)
                ASD.update(self.findSpeechFaceAssociationPartion(asd, posGuides, negGuides))
        else:
            asd, posGuides, negGuides = self.initializeASD(speechKeys)
            ASD.update(self.findSpeechFaceAssociationPartion(asd, posGuides, negGuides)) 
        return ASD
    
    def findSpeechFaceAssociationPartion(self, asd, posGuides, negGuides):
        # sourcery skip: low-code-quality
        if self.verbose:
            negGuidesFaceTracksCount = np.sum(len(faceTracks_) for faceTracks_ in negGuides.values())
            faceTracksCount = np.sum(len(self.speechFaceTracks[key_]['face_tracks']) for key_ in asd.keys())
            print(f'positive guides: {len(posGuides)}/ {len(asd.keys())}')
            print(f'negtive guided face tracks: {negGuidesFaceTracksCount}/{faceTracksCount}')
        # remove keys using the negative guides
        removeKeysCount = 0
        for key_ in asd.keys():
            trackIds = [track[0] for track in self.speechFaceTracks[key_]['face_tracks']\
                        if track[0] not in negGuides[key_]]
            if not len(trackIds):
                del asd[key_]
                removeKeysCount += 1
        
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
        