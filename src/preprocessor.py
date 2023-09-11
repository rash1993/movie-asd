'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-08 12:58:15
 * @modify date 2023-02-08 12:58:15
* @desc [description]
 */'''
import sys, os
from audio_prep.audio_preprocessing import AudioPreProcessor
from video_prep.video_preprocessing import VideoPreProcessor
from local_utils import writeToPickleFile
from ASD.asd_utils import Distances
from TalkNet.TalkNet_wrapper import TalkNetWrapper
import numpy as np
import json

class Preprocessor():
    def __init__(self, videoPath, cacheDir=None, verbose=False, talknet_flag=False):
        self.videoPath = videoPath
        self.videoName = os.path.basename(videoPath)[:-4]
        self.verbose = verbose
        if cacheDir is None:
            cacheDir = '../cache'
        self.cacheDir = cacheDir
        self.talknet_flag = talknet_flag
        os.makedirs(self.cacheDir, exist_ok=True)
        self.audioPrep = AudioPreProcessor(self.videoPath, self.cacheDir, verbose=self.verbose)
        self.videoPrep = VideoPreProcessor(self.videoPath, self.cacheDir, verbose=self.verbose)

    def getTemporallyOverlappingFaceTracks(self, faceTracks, speechSegments):
        
        def getFaceTrackTime(face_track):
            track = sorted(face_track, key=lambda x: x[0])
            return [track[0][0], track[-1][0]]

        faceTrackTimes = [[trackId] + getFaceTrackTime(track) for \
            trackId, track in faceTracks.items()]
        faceTrackTimes.sort(key = lambda x: x[1])
        segmentsOut = {}
        carry = []
        segmentsCounter = 0
        segments = speechSegments
        segments.sort(key = lambda x: x[1])
        for segment in segments:
            segmentId = segment[0]
            segment = segment[1:]
            overlappingFaceTracks = []
            if carry:
                # considering face tracks which starts earlier than the segment and may have overlap.
                newCarry = []
                for faceTrack in carry:
                    overlap = min(faceTrack[2], segment[1]) - max(faceTrack[1], segment[0])
                    if overlap > 0:
                        st = max(faceTrack[1], segment[0])
                        et = min(faceTrack[2], segment[1])
                        overlappingFaceTracks.append([faceTrack[0], st, et])
                    if faceTrack[2] > segment[1]:
                        # keeping the face tracks which end beyond the segment for further segments consideration
                        newCarry.append(faceTrack)
                carry = newCarry
            flag = True
            while flag:
                if len(faceTrackTimes):
                    faceTrack = faceTrackTimes.pop(0)
                    if faceTrack[1] > segment[1]:
                        # if the current face track starts after the segment end
                        # since the face tracks are sorted with starting time we stop here for this segment
                        carry.append(faceTrack)
                        flag = False
                    overlap = min(faceTrack[2], segment[1]) - max(faceTrack[1], segment[0])
                    if overlap > 0:
                        st = max(faceTrack[1], segment[0])
                        et = min(faceTrack[2], segment[1])
                        overlappingFaceTracks.append([faceTrack[0], st, et])
                        if faceTrack[2] > segment[1]:
                            # if the end point of the face track is beyond segment, consider it for the next segment
                            carry.append(faceTrack)
                else:
                    flag = False
            if len(overlappingFaceTracks):
                segmentsOut[segmentId] = {'speech': segment, 'face_tracks': overlappingFaceTracks}
                segmentsCounter += 1
        self.speechFaceTracks = segmentsOut
        speechFaceTracksFile = os.path.join(self.cacheDir, 'speechFaceTracks.pkl')
        writeToPickleFile(self.speechFaceTracks, speechFaceTracksFile)

    def getPotentialActiveSpeakerFaceTracks(self, faceTracks, speechSegments, faceTrackDistances, th=0.4):
        '''
        Method to get potential active speaker face track for each speech segment, which includes all the faces
        in the shot of the speech segment and the faces in the shots which are temporally close to the shot of the
        speech segment. THe face tracks are chosen such that they belong to unique characters.
        '''
        
        self.speechFaceTracksMarginal = {}
        for segment in speechSegments:
            segmentId = segment[0]
            shotId = segmentId.split('_')[0]
            potentialActiveSpeakers = [faceTrackId for faceTrackId, faceTrack in faceTracks.items()\
                                         if faceTrackId.split('_')[0] == shotId]
            marginalFaceTracks = []
            marginalFaceTracks.extend([faceTrackId for faceTrackId, faceTrack in faceTracks.items()\
                                         if int(faceTrackId.split('_')[0]) == int(shotId) + 1])
            marginalFaceTracks.extend([faceTrackId for faceTrackId, faceTrack in faceTracks.items()\
                                            if int(faceTrackId.split('_')[0]) == int(shotId) - 1])
            
            # filter out the face tracks from marginal which are close to face tracks in \
            # potential active speakers in terms of the cosine distance between the face tracks.
            for marginalFaceTrackId in marginalFaceTracks:
                flag = True
                for potentialFaceTrackId in potentialActiveSpeakers:
                    if faceTrackDistances[marginalFaceTrackId][potentialFaceTrackId] < th:
                        flag = False
                        break
                if flag:
                    potentialActiveSpeakers.append(marginalFaceTrackId)
                    
            potentialActiveSpeakers = [[faceTrackId] + segment[1:] for faceTrackId in potentialActiveSpeakers]
            self.speechFaceTracksMarginal[segmentId] = {'speech': segment[1:], 'face_tracks': potentialActiveSpeakers}
        # speechFaceTracksFile = os.path.join(self.cacheDir, 'speechFaceTracks.pkl')
        # writeToPickleFile(self.speechFaceTracks, speechFaceTracksFile)

    def constructGuides(self, talknetscores):
        guides = {}
        for key in self.speechFaceTracks.keys():
            guides[key] = {}
            for faceTrack in self.speechFaceTracks[key]['face_tracks']:
                faceTrackId, st, et = faceTrack
                scores = [score for face, score in \
                            zip(self.videoPrep.faceTracks[faceTrackId],\
                                talknetscores[faceTrackId])\
                            if (face[0] >= st) and (face[0] < et)]
                nonan_scores = [s for s in scores if not np.isnan(float(s))]
                if nonan_scores:
                    guides[key][faceTrackId] = np.mean(nonan_scores)
                else:
                    guides[key][faceTrackId] = 'nan'
        json.dump(guides, open(os.path.join(self.cacheDir, 'talknet_guides.json'), 'w'))
        all_scores = []
        for key, item in talknetscores.items():
            all_scores.extend(item)
        all_scores = [s for s in all_scores if not np.isnan(float(s))]
        posTh = np.percentile(all_scores, 90)
        negTh = np.percentile(all_scores, 40)
        print(f'Talknet posTh: {posTh} negTh: {negTh}')
        self.guides = {'scores': guides, 'posTh': posTh, 'negTh': negTh}

    def prep(self):
        self.audioPrep.run()
        self.videoPrep.run()
        self.getTemporallyOverlappingFaceTracks(self.videoPrep.faceTracks,\
             self.audioPrep.speakerHomoSegments)
        faceTrackDistances = Distances(self.videoPrep.faceTrackFeats, self.audioPrep.speechEmbeddings, \
            self.cacheDir, verbose=self.verbose).faceDistances
        self.getPotentialActiveSpeakerFaceTracks(self.videoPrep.faceTracks,\
             self.audioPrep.speakerHomoSegments, faceTrackDistances)
        if self.talknet_flag:
            talknet = TalkNetWrapper(self.videoPath, self.cacheDir, self.videoPrep.framesObj)
            self.talknetscores = talknet.run(visualization=True)
            self.constructGuides(self.talknetscores)
        else:
            self.guides = None