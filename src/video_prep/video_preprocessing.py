'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-02 16:42:42
 * @modify date 2023-02-02 16:42:42
 * @desc [description]
 */'''
import sys, os, cv2
from src.local_utils import readVideoFrames, writeToPickleFile
import pickle as pkl 
from src.video_prep.deep_face_detector import FaceDetector

class VideoPreProcessor():
    def __init__(self, videoPath, cacheDir, faceDetectorName='retinaface', verbose=False):
        self.videoPath = videoPath
        self.cacheDir = cacheDir
        self.videoName = os.path.basename(videoPath)[:-4]
        self.verbose = verbose
        self.faceDetectorName = faceDetectorName
    
    def getVideoFrames(self, fps=-1, cache=True):
        framesFilePath = os.path.join(self.cacheDir, 'frames.pkl')
        if os.path.isfile(framesFilePath) and cache:
            if self.verbose:
                print('reading frames from the cache')
            framesObj = pkl.load(open(framesFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'reading frames from the video')
            framesObj = readVideoFrames(self.videoPath, fps=fps)
            if cache:
                writeToPickleFile(framesObj, framesFilePath)
        return framesObj

    
    def getFaces(self, frames, cache=True):
        facesFilePath = os.path.join(self.cacheDir, 'faces.pkl')
        if os.path.isfile(facesFilePath) and cache:
            if self.verbose:
                print('reading faces from the cache')
            faces = pkl.load(open(facesFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'extracting faces and saving at: {facesFilePath}')
            faceDetector = FaceDetector(self.faceDetectorName)
            faces = faceDetector.run(frames)
            if cache:
                writeToPickleFile(faces, facesFilePath)        
        return faces

    def getFaceTracks(self, frames, cache=True):
        

        faceTracksFilePath = os.path.join(self.cacheDir, f'face_{self.faceDetectorName}.pkl')
        if os.path.isfile(faceTracksFilePath) and cache:
            if self.verbose:
                print('reading face tracks from cache')
            self.faceTracks = pkl.load(open(faceTracksFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'extracting face tracks and saving at: {faceTracksFilePath}')
            faces = self.getFaces(frames)
            
            if self.faceDetectorName == 'retinaFace':
                self.faceDetector = RetinaFaceWithSortTracker(self.videoPath, \
                                        self.cacheDir, self.framesObj)
            else:
                sys.exit(f'face detector {self.faceDetectorName} not implemented')
            self.faceTracks = self.faceDetector.run()
            writeToPickleFile(self.faceTracks, faceTracksFilePath) 
        
    # def getFaceTrackEmbeddings(self, embeddings='vggface2', faceTracksListName='all'):
    #     if faceTracksListName == 'all':
    #         faceTracksList = list(self.faceTracks.keys())
    #     else:
    #         # TODO: Implement faceTrackList relavent to ASD
    #         sys.exit(f'faceTrackList {faceTracksListName} not implemented')
    #     faceTracksEmbeddingsFile = os.path.join(self.cacheDir, \
    #         f'face_embeddings_{embeddings}_{faceTracksListName}.pkl')
    #     if os.path.isfile(faceTracksEmbeddingsFile):
    #         if self.verbose:
    #             print('reading face track embeddings from cache dir')
    #         self.faceTrackFeats = pkl.load(open(faceTracksEmbeddingsFile, 'rb'))
    #     else:
    #         if self.verbose:
    #             print(f'extracting face track embeddings and saving at: {faceTracksEmbeddingsFile}')
    #         self.embeddingsExtracter = VggFace2Embeddings(self.framesObj, self.faceTracks)
    #         self.faceTrackFeats = self.embeddingsExtracter.extractEmbeddings(faceTracksList)
    #         writeToPickleFile(self.faceTrackFeats, faceTracksEmbeddingsFile)
        
    def visualizeFaceTracks(self, framesObj, faces):
        x_scale = framesObj['width']
        y_scale = framesObj['height']
        for frameNo, boxes in enumerate(faces):
            for box in boxes:
                x1 = int(round(box[0]*x_scale))
                y1 = int(round(box[1]*y_scale))
                x2 = int(round(box[2]*x_scale))
                y2 = int(round(box[3]*y_scale))
                cv2.rectangle(framesObj['frames'][frameNo], (x1, y1),\
                        (x2, y2), color=(255, 0, 0), thickness=2)
        videoSavePath = os.path.join(self.cacheDir, 'face_tracks_sanity_check.mp4')
        print((framesObj['width'], framesObj['height']))
        video_writer = cv2.VideoWriter(videoSavePath, cv2.VideoWriter_fourcc(*'mp4v'), \
            framesObj['fps'], (framesObj['width'], framesObj['height']))
        for frame in framesObj['frames']:
            video_writer.write(frame)
        video_writer.release()
        print(f'face tracks sanity check video saved at: {videoSavePath}')

    
    def run(self):
        framesObj =  self.getVideoFrames(fps=6)
        faces = self.getFaces(framesObj['frames'])
        # self.getFaceTracks()
        # self.getFaceTrackEmbeddings()
        
        ## sanity check face tracks  
        self.visualizeFaceTracks(framesObj, faces)
    
if __name__ == "__main__":
    video_path = '/home/azureuser/cloudfiles/code/tsample3.mp4'
    cache_dir = '/home/azureuser/cloudfiles/code/cache'
    os.makedirs(cache_dir, exist_ok=True)
    videoPreProcessor = VideoPreProcessor(video_path, cache_dir, verbose=True)
    videoPreProcessor.run()
        
            

        