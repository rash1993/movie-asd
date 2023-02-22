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
    def __init__(self, 
                speechFaceTracks, 
                faceTracks, 
                faceFeatures, 
                speechFeatures, cacheDir, guides=None, verbose=False):
        self.speechFaceTracks = speechFaceTracks
        self.faceTracks = faceTracks
        self.faceFeatures = faceFeatures
        self.speechFeatures = speechFeatures
        self.cacheDir = cacheDir
        self.guides = guides
        self.verbose = verbose
        self.similarity = Similarity(measure='correlation')
        self.distanaces = Distances(faceFeatures, speechFeatures, \
            self.cacheDir, verbose=self.verbose)


    def run(self, partitionLen='full'):
        speechFaceAssociation = SpeechFaceAssociation(self.cacheDir,\
                                                      self.speechFaceTracks,\
                                                      self.similarity,\
                                                      self.distanaces,\
                                                      self.faceTracks,\
                                                      self.guides,\
                                                      self.verbose)
        self.asd = speechFaceAssociation.handler(partitionLen)
        asdSaveFile = os.path.join(self.cacheDir, 'asd.pkl')
        writeToPickleFile(self.asd, asdSaveFile)
    
    def visualizeASD(self, videoPath):
        framesFile = os.path.join(self.cacheDir, 'frames.pkl')
        if os.path.isfile(framesFile):
            framesObj = pkl.load(open(framesFile, 'rb'))
        else:
            framesObj = readVideoFrames(videoPath)

        frames = framesObj['frames']
        for _, faceTrack in self.faceTracks.items():
            for box in faceTrack:
                frameNo = int(round(box[0]*framesObj['fps']))
                if frameNo < len(frames):
                    x1 = int(round(box[1]*framesObj['width']))
                    y1 = int(round(box[2]*framesObj['height']))
                    x2 = int(round(box[3]*framesObj['width']))
                    y2 = int(round(box[4]*framesObj['height']))
                    cv2.rectangle(frames[frameNo], (x1, y1), (x2, y2), (255, 0, 0))

        for key, faceTrackId in self.asd.items():
            st, et = self.speechFaceTracks[key]['speech']
            faceTrack = self.faceTracks[faceTrackId]
            for box in faceTrack:
                if box[0] >= st and box[0] <= et:
                    frameNo = int(round(box[0]*framesObj['fps']))
                    if frameNo < len(frames):
                        x1 = int(round(box[1]*framesObj['width']))
                        y1 = int(round(box[2]*framesObj['height']))
                        x2 = int(round(box[3]*framesObj['width']))
                        y2 = int(round(box[4]*framesObj['height']))
                        cv2.rectangle(frames[frameNo], (x1, y1), (x2, y2), (0, 255, 0))

        videoName = os.path.basename(videoPath)[:-4]
        videoSavePath = os.path.join(self.cacheDir, f'{videoName}_asdOut.mp4')
        wavPath = os.path.join(self.cacheDir, 'audio.wav')
        if not os.path.isfile(wavPath):
            wavCmd = f'ffmpeg -y -nostdin -loglevel error -y -i {videoPath} \
                -ar 16k -ac 1 {wavPath}'
            subprocess.call(wavCmd, shell=True, stdout=False)

        make_video(frames, framesObj['fps'], videoSavePath, sound_fname=wavPath,
                    keep_aud_file=True)
