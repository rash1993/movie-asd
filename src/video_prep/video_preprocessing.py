'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-02 16:42:42
 * @modify date 2023-02-02 16:42:42
 * @desc [description]
 */'''
import sys, os, cv2
sys.path.append('../')
sys.path.append('../../')
from local_utils import readVideoFrames, writeToPickleFile
import pickle as pkl 
from video_prep.retina_face import RetinaFaceWithSortTracker
from utils_cams import make_video
from video_prep.vggFace2_embeddings import VggFace2Embeddings


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
        
    def visualizeFaceTracks(self):
        for trackID, boxes in self.faceTracks.items():  
            for box in boxes:
                frameNo = int(round(box[0]*self.framesObj['fps']))
                if frameNo < len(self.framesObj['frames']):
                    x1 = int(round(box[1]*self.framesObj['width']))
                    y1 = int(round(box[2]*self.framesObj['height']))
                    x2 = int(round(box[3]*self.framesObj['width']))
                    y2 = int(round(box[4]*self.framesObj['height']))
                    cv2.rectangle(self.framesObj['frames'][frameNo], (x1, y1),\
                            (x2, y2), color=(255, 0, 0), thickness=2)
        videoSavePath = os.path.join(self.cacheDir, 'face_tracks_sanity_check.mp4')
        make_video(self.framesObj['frames'], self.framesObj['fps'], videoSavePath)

    
    def run(self):
        self.getVideoFrames()
        self.getFaceTracks()
        self.getFaceTrackEmbeddings()
        
        ## sanity check face tracks  
        # self.visualizeFaceTracks()
        
            

        