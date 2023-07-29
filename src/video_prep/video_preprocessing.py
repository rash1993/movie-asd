'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-02 16:42:42
 * @modify date 2023-02-02 16:42:42
 * @desc [description]
 */'''

from src.local_utils import readVideoFrames, writeToPickleFile, timeCode2seconds
from src.video_prep.deep_face_detector import FaceDetector
from src.video_prep.sort_tracker import SortTracker
from scenedetect import detect, AdaptiveDetector, ContentDetector
import pickle as pkl 
import sys, os, cv2
import numpy as np
from tqdm import tqdm

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

    
    def getFaces(self, framesObj, cache=True):
        facesFilePath = os.path.join(self.cacheDir, 'faces.pkl')
        if os.path.isfile(facesFilePath) and cache:
            if self.verbose:
                print('reading faces from the cache')
            faces_all = pkl.load(open(facesFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'extracting faces and saving at: {facesFilePath}')
            faceDetector = FaceDetector(self.faceDetectorName)
            faces_all = faceDetector.run(framesObj['frames'])
            # add time stamp to each face
            for i, faces_frame in enumerate(faces_all):
                time_stamp = i / framesObj['fps']
                for bbox in faces_frame:
                    bbox.insert(0, time_stamp)
            if cache:
                writeToPickleFile(faces_all, facesFilePath)        
        return faces_all

    def getFaceTracks(self, framesObj, tracker='sort', cache=True):
        faceTracksFilePath = os.path.join(self.cacheDir, f'face_tracks_{self.faceDetectorName}_{tracker}.pkl')
        if os.path.isfile(faceTracksFilePath) and cache:
            if self.verbose:
                print('reading face tracks from cache')
            face_tracks = pkl.load(open(faceTracksFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'extracting face tracks and saving at: {faceTracksFilePath}')
            # get faces
            faces =  self.getFaces(framesObj, cache=cache)

            # getshots
            scenes = detect(self.videoPath, ContentDetector())
            scenes = [[timeCode2seconds(scene[0].get_timecode()),  \
                       timeCode2seconds(scene[1].get_timecode())] for scene in scenes]
            
            # get_face_tracks
            face_tracks = {}
            # face_tracks = SortTracker().track(faces)
            for i, scene in tqdm(enumerate(scenes), desc='extracting face tracks'):
                tracks_prefix = str(i)
                start_time, end_time = scene
                start_frame = int(np.floor(start_time * framesObj['fps']))
                end_frame = int(np.ceil(end_time * framesObj['fps']))
                if start_frame == end_frame:
                    continue
                dets_scene =  faces[start_frame:end_frame]
                face_tracks.update(SortTracker(tracks_prefix).track(dets_scene))
            # filter face tracks to remove short tracks > 0.2sec
            face_tracks = {track_id: track for track_id, track in face_tracks.items() if len(track) > 0.2 * framesObj['fps']}
            if cache:
                writeToPickleFile(face_tracks, faceTracksFilePath)
        return face_tracks
        
    # def visualizeFaceTracks(self, framesObj, faces):
    #     x_scale = framesObj['width']
    #     y_scale = framesObj['height']
    #     for frameNo, boxes in enumerate(faces):
    #         for box in boxes:
    #             x1 = int(round(box[0]*x_scale))
    #             y1 = int(round(box[1]*y_scale))
    #             x2 = int(round(box[2]*x_scale))
    #             y2 = int(round(box[3]*y_scale))
    #             cv2.rectangle(framesObj['frames'][frameNo], (x1, y1),\
    #                     (x2, y2), color=(255, 0, 0), thickness=2)
    #     videoSavePath = os.path.join(self.cacheDir, 'face_tracks_sanity_check.mp4')
    #     print((framesObj['width'], framesObj['height']))
    #     video_writer = cv2.VideoWriter(videoSavePath, cv2.VideoWriter_fourcc(*'mp4v'), \
    #         framesObj['fps'], (framesObj['width'], framesObj['height']))
    #     for frame in framesObj['frames']:
    #         video_writer.write(frame)
    #     video_writer.release()
    #     print(f'face tracks sanity check video saved at: {videoSavePath}')

    def visualizeFaceTracks(self, framesObj, facetracks):
        '''
        method to visualize face tracks on the video
        
        Args:
            framesObj (dict): dictionary containing video frames and fps
            faces (list): list of faces detected in each frame
            facetracks (dict): dictionary containing face tracks
        '''
        x_scale = framesObj['width']
        y_scale = framesObj['height']
        frames = framesObj['frames']
        for track_id, track in facetracks.items():
            for box in track:
                x1 = int(round(box[1]*x_scale))
                y1 = int(round(box[2]*y_scale))
                x2 = int(round(box[3]*x_scale))
                y2 = int(round(box[4]*y_scale))
                frame_no = int(round(box[0]*framesObj['fps']))
                cv2.rectangle(frames[frame_no], (x1, y1),\
                        (x2, y2), color=(255, 0, 0), thickness=2)
                cv2.putText(frames[frame_no], str(track_id), (x1, y1),\
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        videoSavePath = os.path.join(self.cacheDir, 'face_tracks_sanity_check.mp4')
        video_writer = cv2.VideoWriter(videoSavePath, cv2.VideoWriter_fourcc(*'mp4v'), \
            framesObj['fps'], (framesObj['width'], framesObj['height']))
        for frame in framesObj['frames']:
            video_writer.write(frame)
        video_writer.release()
        print(f'face tracks sanity check video saved at: {videoSavePath}')

    def run(self):
        framesObj = self.getVideoFrames(fps=6)
        # faces = self.getFaces(framesObj)
        face_tracks = self.getFaceTracks(framesObj)
        # self.getFaceTracks()
        # self.getFaceTrackEmbeddings()
        
        ## sanity check face tracks  
        self.visualizeFaceTracks(framesObj, face_tracks)
    
if __name__ == "__main__":
    video_path = '/home/azureuser/cloudfiles/code/tsample3.mp4'
    cache_dir = '/home/azureuser/cloudfiles/code/cache'
    os.makedirs(cache_dir, exist_ok=True)
    videoPreProcessor = VideoPreProcessor(video_path, cache_dir, verbose=True)
    videoPreProcessor.run()
        
            

        