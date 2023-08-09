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

class Preprocessor():
    def __init__(self, videoPath, cacheDir=None, verbose=False):
        self.videoPath = videoPath
        self.videoName = os.path.basename(videoPath)[:-4]
        self.verbose = verbose
        if cacheDir is None:
            cacheDir = '../cache'
        self.cacheDir = cacheDir
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
        speechFaceTracksFile = os.path.join(self.cacheDir, 'speechFaceTracks.pkl')
        writeToPickleFile(segmentsOut, speechFaceTracksFile)
        return segmentsOut

    def prep(self, fps, resolution):
        speechEmbeddings = self.audioPrep.run()
        faceTrackEmbeddings = self.videoPrep.run(fps, resolution)
        faceTracks = {track_id: faceTrack['track'] for track_id, faceTrack in faceTrackEmbeddings.items()}
        speechSegments = [[segment_id] + segment['segment'] for segment_id, segment in speechEmbeddings.items()]
        speechFaceTracks = self.getTemporallyOverlappingFaceTracks(faceTracks, speechSegments)
        return speechFaceTracks, faceTrackEmbeddings, speechEmbeddings

