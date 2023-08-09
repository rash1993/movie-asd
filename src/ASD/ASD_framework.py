'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-08 12:43:05
 * @modify date 2023-02-08 12:43:05
 * @desc [description]
 */'''

import sys, os, cv2, subprocess
import pickle as pkl
from ASD.asd_utils import Distances, Similarity
from ASD.speech_face_association import SpeechFaceAssociation
from local_utils import readVideoFrames, writeToPickleFile
from utils_cams import make_video



class ASD():
    def __init__(self, cache_dir, similarity_mesaure='correlation', verbose=False):
        self.cacheDir = cache_dir
        self.similarity = Similarity(measure=similarity_mesaure)
        self.verbose = verbose

    def offscreenSpeakercorrection(self, th=0.1):
        speechKeys = list(self.asd.keys())
        audioDistances = self.distances.computeDistanceMatrix(\
                                    speechKeys, modality='speech')
        faceDistances = self.distances.computeDistanceMatrix(\
                                    speechKeys, asd=self.asd, modality='face')
        corr = self.similarity.computeAvgSimilarity(\
                                    audioDistances, faceDistances, avg=False)
        offScreenSpeechKeys = [key_ for key_, corr_ in zip(speechKeys, corr) if corr_ < th]
        self.asd = {key:self.asd[key] for key in self.asd.keys() if key not in offScreenSpeechKeys}
        
    def run(self, speechFaceTracks, 
            faceTrackEmbeddings, 
            speechEmbeddings,
            guides=None, 
            partitionLen='full'):
        self.distances = Distances(faceTrackEmbeddings, speechEmbeddings, self.cacheDir, self.verbose)
        self.faceTracks = {track_id: track['track'] for track_id, track in faceTrackEmbeddings.items()}
        self.sppeechSegments = {segment_id: segment['segment'] for segment_id, segment in speechEmbeddings.items()}
        speechFaceAssociation = SpeechFaceAssociation(self.cacheDir,\
                                                      speechFaceTracks,\
                                                      self.similarity,\
                                                      self.distances,\
                                                      self.faceTracks,\
                                                      guides,\
                                                      self.verbose)
        self.asd = speechFaceAssociation.handler(partitionLen)
        self.offscreenSpeakercorrection()
        asdSaveFile = os.path.join(self.cacheDir, 'asd.pkl')
        asd_checkpoint = self.createCheckpoint(self.asd, self.sppeechSegments, self.faceTracks)
        writeToPickleFile(asd_checkpoint, asdSaveFile)
        return asd_checkpoint
    
    def createCheckpoint(self, asd, speechSegments, faceTracks):
        asd_checkpoint = {}
        for segment_id, active_face_track_id in asd.items():
            segment = speechSegments[segment_id]
            active_face_track = faceTracks[active_face_track_id]
            active_face_track = [box for box in active_face_track if box[0] >= segment[0] and box[0] <= segment[1]]
            asd_checkpoint[segment_id] = {'track_id': active_face_track_id, 'track':active_face_track}
        return asd_checkpoint
