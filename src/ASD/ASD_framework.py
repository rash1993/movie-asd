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
from local_utils import readVideoFrames, writeToPickleFile, getFaceProfile, getEyeDistance
from utils_cams import make_video
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm




class ASD():
    def __init__(self, 
                speechFaceTracks, 
                faceTracks, 
                faceFeatures, 
                speechFeatures, cacheDir, guides=None, verbose=False, marginalFaceTracks=None):
        self.speechFaceTracks = speechFaceTracks
        self.faceTracks = faceTracks
        self.faceFeatures = faceFeatures
        self.speechFeatures = speechFeatures
        self.cacheDir = cacheDir
        self.guides = guides
        self.verbose = verbose
        self.marginalFaceTracks = marginalFaceTracks
        self.similarity = Similarity(measure='correlation')
        self.distances = Distances(faceFeatures, speechFeatures, \
            self.cacheDir, verbose=self.verbose)

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
    
    def getMarginalDistances(self, currCorr):
        keys = list(self.asd.keys())
        audioDistances = self.distances.computeDistanceMatrix(\
                                        keys, modality='speech') 
        marginal_distances = []
        for i, key in tqdm(enumerate(keys), desc='computing marginal distances'):
            audioDistanceVector = audioDistances[i]
            currCorri = currCorr[i]
            mdistances = []
            for trackId  in self.marginalFaceTracks[key]['face_tracks']:
                if trackId[0] != self.asd[key]:
                    asd = self.asd.copy()
                    asd[key] = trackId[0]
                    keys_ = keys
                    faceDistances = self.distances.computeDistanceMatrix(\
                                        keys_, asd=asd, modality='face')
                    faceDistanceVector = faceDistances[i]
                    newCorri = pearsonr(audioDistanceVector, faceDistanceVector)[0]
                    mdistances.append([trackId[0], newCorri - currCorri])
            mdistances.sort(key=lambda x: x[1], reverse=True)
            marginal_distances.append([key] + mdistances[0])
        return marginal_distances

    def offscreenSpeakercorrection2(self):
        # optimize the matrix to the maximum value by replacing the faces with the neighboring \
        # faces in the matrix. Starting with the highly likely off-screen speaker move to the least likely
        last_corr = 0.0
        audioDistances = self.distances.computeDistanceMatrix(keys=self.asd.keys(), modality='speech')
        faceDistances = self.distances.computeDistanceMatrix(keys=self.asd.keys(), asd=self.asd, modality='face')
        last_corr = self.similarity.computeAvgSimilarity(audioDistances, faceDistances, avg=False)
        curr_corr = last_corr
        while True:
            last_corr = curr_corr.copy()
            marginal_distances = self.getMarginalDistances(curr_corr)
            sel_key = max(marginal_distances, key=lambda x: x[2])
            if (sel_key[2] > 0) and (sel_key[2] > 0.1):
                self.asd[sel_key[0]] = sel_key[1]
                faceDistances = self.distances.computeDistanceMatrix(keys=self.asd.keys(), asd=self.asd, modality='face')
                curr_corr = self.similarity.computeAvgSimilarity(audioDistances, faceDistances, avg=False)
                print(f'curr_corr: {np.mean(curr_corr)} | sel_key: {sel_key}, last_corr: {np.mean(last_corr)}')
            else:
                break
            if (np.mean(curr_corr) - np.mean(last_corr))/np.mean(last_corr) < 0.01:
                break
            

    def run(self, partitionLen='full'):
        speechFaceAssociation = SpeechFaceAssociation(self.cacheDir,\
                                                      self.speechFaceTracks,\
                                                      self.similarity,\
                                                      self.distances,\
                                                      self.faceTracks,\
                                                      self.guides,\
                                                      self.verbose)
        self.asd = speechFaceAssociation.handler(partitionLen)
        # self.offscreenSpeakercorrection2()
        self.offscreenSpeakercorrection()
        audioDistances = self.distances.computeDistanceMatrix(keys=self.asd.keys(), modality='speech')
        faceDistances = self.distances.computeDistanceMatrix(keys=self.asd.keys(), asd=self.asd, modality='face')
        corr = self.similarity.computeAvgSimilarity(audioDistances, faceDistances, avg=False)
        print([[key, corr_] for key, corr_ in zip(self.asd.keys(), corr)])
        asdSaveFile = os.path.join(self.cacheDir, 'asd.pkl')
        writeToPickleFile(self.asd, asdSaveFile)
    
    def visualizeDistanceMatrices(self):
        speechKeys = list(self.asd.keys())
        speechKeys.sort(key=lambda x: self.speechFaceTracks[x]['speech'][0])
        audioDistances = self.distances.computeDistanceMatrix(\
                                    speechKeys, modality='speech')
        faceDistances = self.distances.computeDistanceMatrix(\
                                    speechKeys, asd=self.asd, modality='face')
        fig, ax = plt.subplots(1,2)
    
        ax[0].imshow(audioDistances)
        ax[1].imshow(faceDistances)
        plt.savefig(os.path.join(self.cacheDir, 'distance_matrix.png'), dpi=300)

    def visualizeASD(self, videoPath):
        framesFile = os.path.join(self.cacheDir, 'frames.pkl')
        if os.path.isfile(framesFile):
            framesObj = pkl.load(open(framesFile, 'rb'))
        else:
            framesObj = readVideoFrames(videoPath)

        frames = framesObj['frames']

        for facetrackId, faceTrack in self.faceTracks.items():
            for box in faceTrack:
                frameNo = int(round(box[0]*framesObj['fps']))
                if frameNo < len(frames):
                    x1 = int(round(box[1]*framesObj['width']))
                    y1 = int(round(box[2]*framesObj['height']))
                    x2 = int(round(box[3]*framesObj['width']))
                    y2 = int(round(box[4]*framesObj['height']))
                    cv2.rectangle(frames[frameNo], (x1, y1), (x2, y2), (255, 0, 0))
                    # printing the name of the face track
                    cv2.putText(frames[frameNo], str(facetrackId), (x1, y1 - 10), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    # draw markers at face landmarks
                    landms = np.array(box[-1]).reshape((-1, 2))
                    # faceProfile = getFaceProfile(landms)
                    landms[:,0]*=framesObj['width']
                    landms[:,1]*=framesObj['height']
                    eyeDistance = str(round(getEyeDistance(landms) , 2))
                    cv2.putText(frames[frameNo], eyeDistance, (x2, y2 + 10), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    for i, mark in enumerate(landms):
                        x = int(mark[0])
                        y = int(mark[1])
                        cv2.drawMarker(frames[frameNo], (x, y), \
                                color=(0,0,255), thickness=1, markerType=cv2.MARKER_CROSS, \
                                markerSize=4)
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
        

        # printing the active speaker face track id on the frames
        for key, faceTrackId in self.asd.items():
            st, et = self.speechFaceTracks[key]['speech']
            sf = int(round(st*framesObj['fps']))
            ef = int(round(et*framesObj['fps']))
            for frameNo in range(sf, ef):
                if frameNo < len(frames):
                    cv2.putText(frames[frameNo], str(faceTrackId), (10, 10), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(frames[frameNo], str(key), \
                                (int(framesObj['width'])-50, int(framesObj['height'])-20), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        videoName = os.path.basename(videoPath)[:-4]
        videoSavePath = os.path.join(self.cacheDir, f'{videoName}_asdOut.mp4')
        wavPath = os.path.join(self.cacheDir, 'audio.wav')
        if not os.path.isfile(wavPath):
            wavCmd = f'ffmpeg -y -nostdin -loglevel error -y -i {videoPath} \
                -ar 16k -ac 1 {wavPath}'
            subprocess.call(wavCmd, shell=True, stdout=False)
        
        video_writer = cv2.VideoWriter(videoSavePath, cv2.VideoWriter_fourcc(*'mp4v'), \
                                   framesObj['fps'], (int(framesObj['width']), int(framesObj['height'])))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        videoSavePathTmp = os.path.join(self.cacheDir, f'{videoName}_asdOut_tmp.mp4')
        audio_video_merge_cmd  = f'ffmpeg -i {videoSavePath} -i {wavPath} -c:v copy -c:a aac {videoSavePathTmp}'
        subprocess.call(audio_video_merge_cmd, shell=True, stdout=False)
        os.rename(f'{videoSavePathTmp}', videoSavePath)
        # TODO: remove the tmp file
        print(f'asd video saved at {videoSavePath}')
