'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-02 16:42:42
 * @modify date 2023-02-02 16:42:42
 * @desc [description]
 */'''
import sys, os, cv2, json
import pickle as pkl 
import numpy as np
sys.path.append('../')
sys.path.append('../../')
from local_utils import readVideoFrames, writeToPickleFile, inside
from tqdm import tqdm
from video_prep.retina_face import RetinaFaceWithSortTracker
from video_prep.vggFace2_embeddings import VggFace2Embeddings
from video_prep.yolov8_with_sort import YoloWithSortTracker
from video_prep.dinov2_embeddings import DinoV2Embeddings

class VideoPreProcessor():
    def __init__(self, videoPath, cacheDir, faceDetectorName='retinaFace', verbose=False):
        self.videoPath = videoPath
        self.cacheDir = cacheDir
        self.videoName = os.path.basename(videoPath)[:-4]
        self.verbose = verbose
        self.faceDetectorName = faceDetectorName
    
    def getVideoFrames(self):
        framesFilePath = os.path.join(self.cacheDir, 'frames.pkl')
        if os.path.isfile(framesFilePath):
            if self.verbose:
                print('reading frames form the cache')
            self.framesObj = pkl.load(open(framesFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'reading frames from the video and saving as: {framesFilePath}')
            self.framesObj = readVideoFrames(self.videoPath)
            writeToPickleFile(self.framesObj, framesFilePath)
    
    def getFaceTracks(self):
        faceTracksFilePath = os.path.join(self.cacheDir, f'face_{self.faceDetectorName}.pkl')
        if os.path.isfile(faceTracksFilePath):
            if self.verbose:
                print('reading face tracks from cache')
            self.faceTracks = pkl.load(open(faceTracksFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'extracting face tracks and saving at: {faceTracksFilePath}')
            if self.faceDetectorName == 'retinaFace':
                self.faceDetector = RetinaFaceWithSortTracker(self.videoPath, \
                                        self.cacheDir, self.framesObj)
            else:
                sys.exit(f'face detector {self.faceDetectorName} not implemented')
            self.faceTracks = self.faceDetector.run()
            writeToPickleFile(self.faceTracks, faceTracksFilePath) 

    def getFaceTrackEmbeddings(self, embeddings='vggface2', faceTracksListName='all'):
        if faceTracksListName == 'all':
            faceTracksList = list(self.faceTracks.keys())
        else:
            # TODO: Implement faceTrackList relavent to ASD
            sys.exit(f'faceTrackList {faceTracksListName} not implemented')
        faceTracksEmbeddingsFile = os.path.join(self.cacheDir, \
            f'face_embeddings_{embeddings}_{faceTracksListName}.pkl')
        if os.path.isfile(faceTracksEmbeddingsFile):
            if self.verbose:
                print('reading face track embeddings from cache dir')
            self.faceTrackFeats = pkl.load(open(faceTracksEmbeddingsFile, 'rb'))
        else:
            if self.verbose:
                print(f'extracting face track embeddings and saving at: {faceTracksEmbeddingsFile}')
            self.embeddingsExtracter = VggFace2Embeddings(self.framesObj, self.faceTracks)
            self.faceTrackFeats = self.embeddingsExtracter.extractEmbeddings(faceTracksList)
            writeToPickleFile(self.faceTrackFeats, faceTracksEmbeddingsFile)
    
    def getBodyTracks(self):
        bodyTracksFilePath = os.path.join(self.cacheDir, f'body_tracks.pkl')
        if os.path.isfile(bodyTracksFilePath):
            if self.verbose:
                print('reading body tracks from cache')
            self.bodyTracks = pkl.load(open(bodyTracksFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'extracting body tracks and saving at: {bodyTracksFilePath}')
            self.bodyDetector = YoloWithSortTracker(self.videoPath, \
                                                    self.cacheDir,\
                                                    self.framesObj)
            self.bodyTracks = self.bodyDetector.run()
            writeToPickleFile(self.bodyTracks, bodyTracksFilePath)
    
    def combineFaceTracks(self, faceTracksGroup):
        # print(len(faceTracksGroup), len(self.faceTracks.keys()), len(self.faceBodyMap.keys()))
        # combine the faces of the tracks
        bodyTrack = self.faceBodyMap[faceTracksGroup[0]]
        combined = []
        for faceTrack in faceTracksGroup:
            combined.extend(self.faceTracks[faceTrack])
            self.faceTracks.pop(faceTrack)
            self.faceBodyMap.pop(faceTrack)
        self.faceTracks[faceTracksGroup[0]] = combined
        self.faceBodyMap[faceTracksGroup[0]] = bodyTrack
        # print(len(self.faceTracks.keys()), len(self.faceBodyMap.keys()))

    def mapBodyFace(self):

        def getNumOverlaps(faceTrack, bodyTrack):
            faceTrack = {int(round(b[0]*self.framesObj['fps'])): b for b in faceTrack}
            bodyTrack = {int(round(b[0]*self.framesObj['fps'])): b for b in bodyTrack}
            numOverlaps = 0
            for ts, faceBox in faceTrack.items():
                if ts in bodyTrack.keys():
                    numOverlaps += inside(faceBox[1:5], bodyTrack[ts][1:5]) 
            return numOverlaps
        shotWiseFaceTracks = {}
        shotWiseBodyTrack = {}
        self.faceBodyMap = {} # for every face there will be body track

        for faceTrack in self.faceTracks.keys():
            shotId = faceTrack.split('_')[0]
            if shotId in shotWiseFaceTracks.keys():
                shotWiseFaceTracks[shotId].append(faceTrack)
            else:
                shotWiseFaceTracks[shotId] = [faceTrack]
        
        for bodyTrack in self.bodyTracks.keys():
            shotId = bodyTrack.split('_')[0]
            if shotId in shotWiseBodyTrack.keys():
                shotWiseBodyTrack[shotId].append(bodyTrack)
            else:
                shotWiseBodyTrack[shotId] = [bodyTrack]

        # TODO: Combine body tracks when they have same face track associated
        # In each shot, assigning a bodyTrack to each FaceTrack
        for shotId, faceTracks in tqdm(shotWiseFaceTracks.items()):
            if shotId not in shotWiseBodyTrack.keys():
                continue
            bodyTracks = shotWiseBodyTrack[shotId]
            for faceTrack in faceTracks:
                overlaps = []
                for bodyTrack in bodyTracks:
                    overlaps.append([bodyTrack, getNumOverlaps(self.faceTracks[faceTrack], \
                                                               self.bodyTracks[bodyTrack])])
                self.faceBodyMap[faceTrack] = max(overlaps, key=lambda x:x[1])[0]
        
        # combine the face tracks which are associated with same body tracks. 
        for shotId, faceTracks in shotWiseFaceTracks.items():
            if shotId in shotWiseBodyTrack.keys():
                bodyFaceTracksMaps = {}
                for faceTrack in faceTracks:
                    bodyTrack = self.faceBodyMap[faceTrack]
                    if bodyTrack in bodyFaceTracksMaps.keys():
                        bodyFaceTracksMaps[bodyTrack].append(faceTrack)
                    else:
                        bodyFaceTracksMaps[bodyTrack] = [faceTrack]
                for faceTracksGroup in bodyFaceTracksMaps.values():
                    if len(faceTracksGroup) > 1:
                        self.combineFaceTracks(faceTracksGroup)

        
        faceBodyMapFile = os.path.join(self.cacheDir, 'face_bodyMap.json')
        with open(faceBodyMapFile, 'w') as fo:
            json.dump(self.faceBodyMap, fo)
        

        # remove the unassociated body tracks
        for trackId in self.bodyTracks.keys():
            if trackId not in self.faceBodyMap.values():
                self.bodyTracks[trackId].pop()
        
    def getBodyTrackEmbeddings(self):
        bodyTrackEmbeddingsFile = os.path.join(self.cacheDir, 'bodyTracksEmbeddings.pkl')
        if os.path.isfile(bodyTrackEmbeddingsFile):
            if self.verbose:
                print('reading bodyTracks embeddings from cache')
            self.bodyTracksEmbeddings = pkl.load(open(bodyTrackEmbeddingsFile, 'rb'))
        else:
            if self.verbose:
                print(f'computing bodyTracks embeddings and savign at {bodyTrackEmbeddingsFile}')
            self.bodyTracksEmbeddings = DinoV2Embeddings().run(self.framesObj, self.bodyTracks)
            writeToPickleFile(self.bodyTracksEmbeddings, bodyTrackEmbeddingsFile)

    def visualizeFaceTracks(self):
        for trackID, boxes in self.bodyTracks.items():  
            for box in boxes:
                frameNo = int(round(box[0]*self.framesObj['fps']))
                if frameNo < len(self.framesObj['frames']):
                    x1 = int(round(box[1]*self.framesObj['width']))
                    y1 = int(round(box[2]*self.framesObj['height']))
                    x2 = int(round(box[3]*self.framesObj['width']))
                    y2 = int(round(box[4]*self.framesObj['height']))
                    cv2.rectangle(self.framesObj['frames'][frameNo], (x1, y1),\
                            (x2, y2), color=(255, 255, 0), thickness=2)
                    cv2.putText(self.framesObj['frames'][frameNo], str(trackID), \
                                    (x1+10, y1+10), \
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)        
                    
        for trackID, boxes in self.faceTracks.items():  
            color = list(np.random.random(size=3) * 256)
            for box in boxes:
                frameNo = int(round(box[0]*self.framesObj['fps']))
                if frameNo < len(self.framesObj['frames']):
                    x1 = int(round(box[1]*self.framesObj['width']))
                    y1 = int(round(box[2]*self.framesObj['height']))
                    x2 = int(round(box[3]*self.framesObj['width']))
                    y2 = int(round(box[4]*self.framesObj['height']))
                    cv2.rectangle(self.framesObj['frames'][frameNo], (x1, y1),\
                            (x2, y2), color=color, thickness=2)
                    cv2.putText(self.framesObj['frames'][frameNo], str(trackID), \
                                    (int((x2+x1)/2), int((y2+y1)/2)), \
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for box in self.bodyTracks[self.faceBodyMap[trackID]]:
                frameNo = int(round(box[0]*self.framesObj['fps']))
                if frameNo < len(self.framesObj['frames']):
                    x1 = int(round(box[1]*self.framesObj['width']))
                    y1 = int(round(box[2]*self.framesObj['height']))
                    x2 = int(round(box[3]*self.framesObj['width']))
                    y2 = int(round(box[4]*self.framesObj['height']))
                    cv2.rectangle(self.framesObj['frames'][frameNo], (x1, y1),\
                            (x2, y2), color=color, thickness=2)
                    cv2.putText(self.framesObj['frames'][frameNo], str(self.faceBodyMap[trackID]), \
                                    (x1+10, y1+10), \
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    

        videoSavePath = os.path.join(self.cacheDir, 'face_tracks_sanity_check.mp4')
        video_writer = cv2.VideoWriter(videoSavePath, cv2.VideoWriter_fourcc(*'mp4v'), \
                                   self.framesObj['fps'], (int(self.framesObj['width']), int(self.framesObj['height'])))
        for frame in self.framesObj['frames']:
            video_writer.write(frame)
        video_writer.release()
    
    def run(self):
        self.getVideoFrames()
        self.getFaceTracks()
        self.getBodyTracks()
        self.mapBodyFace()
        self.getFaceTrackEmbeddings()
        self.getBodyTrackEmbeddings()
        ## sanity check face tracks  
        self.visualizeFaceTracks()
        
            

        