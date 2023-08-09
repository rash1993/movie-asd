'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-02 16:42:42
 * @modify date 2023-02-02 16:42:42
 * @desc [description]
 */'''

from src.local_utils import readVideoFrames, writeToPickleFile, timeCode2seconds, split_face_tracks
from src.video_prep.deep_face_detector import FaceDetector
from src.video_prep.sort_tracker import SortTracker
from scenedetect import detect, AdaptiveDetector, ContentDetector
import pickle as pkl 
import sys, os, cv2, csv
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from random import shuffle

class VideoPreProcessor():
    def __init__(self, videoPath, cacheDir, faceDetectorName='retinaface', verbose=False):
        self.videoPath = videoPath
        self.cacheDir = cacheDir
        self.videoName = os.path.basename(videoPath)[:-4]
        self.verbose = verbose
        self.faceDetectorName = faceDetectorName
    
    def getVideoFrames(self, fps=-1, res=None, cache=True):
        framesFilePath = os.path.join(self.cacheDir, 'frames.pkl')
        if os.path.isfile(framesFilePath) and cache:
            if self.verbose:
                print('reading frames from the cache')
            framesObj = pkl.load(open(framesFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'reading frames from the video')
            framesObj = readVideoFrames(self.videoPath, fps=fps, res=res)
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
            del faceDetector    
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
            with open(os.path.join(self.cacheDir, 'scenes.csv'), 'w') as f:
                csv_writer = csv.writer(f)
                for line in scenes:
                    csv_writer.writerow(line)
            # get_face_tracks
            face_tracks = {}
            for scene_idx, scene in enumerate(scenes):
                start_frame = int(np.ceil(scene[0] * framesObj['fps']))
                end_frame = int(np.floor(scene[1] * framesObj['fps']))
                scene_faces = faces[start_frame:end_frame]
                if tracker == 'sort':
                    scene_face_tracks = SortTracker(prefix=scene_idx).track(scene_faces)
                else:
                    raise NotImplementedError
                face_tracks.update(scene_face_tracks)
            # filter face tracks to remove short tracks > 0.2sec
            face_tracks = {track_id: track for track_id, track in face_tracks.items() if len(track) > 0.2 * framesObj['fps']}
            if cache:
                writeToPickleFile(face_tracks, faceTracksFilePath)
        return face_tracks
    
    def getFaceTrackEmbeddings(self, framesObj, faceTracks, cache=True):
        faceTrackEmbeddingsFilePath = os.path.join(self.cacheDir, f'face_track_embeddings_{self.faceDetectorName}.pkl')
        if os.path.isfile(faceTrackEmbeddingsFilePath) and cache:
            if self.verbose:
                print('reading face track embeddings from cache')
            face_track_embeddings = pkl.load(open(faceTrackEmbeddingsFilePath, 'rb'))
        else:
            if self.verbose:
                print(f'extracting face track embeddings and saving at: {faceTrackEmbeddingsFilePath}')
            face_track_embeddings = {}
            for track_id, track in tqdm(faceTracks.items(), desc='extracting face track embeddings'):
                face_track_embeddings[track_id] = []
                # select 4 bbox from each track at random without replacement
                shuffle(track)
                track = track[:4]
                # get bbox crop for each image in the track
                for box in track:
                    frame_no = int(round(box[0]*framesObj['fps']))
                    x1 = int(round(box[1]*framesObj['width']))
                    y1 = int(round(box[2]*framesObj['height']))
                    x2 = int(round(box[3]*framesObj['width']))
                    y2 = int(round(box[4]*framesObj['height']))
                    face_ = framesObj['frames'][frame_no][y1:y2, x1:x2]
                    if face_.shape[0] == 0 or face_.shape[1] == 0:
                        continue
                    embedd_ = DeepFace.represent(face_, model_name='VGG-Face', align=True, detector_backend='skip')[0]['embedding']
                    face_track_embeddings[track_id].append(embedd_)
                if len(face_track_embeddings[track_id]) == 0:
                    del face_track_embeddings[track_id]
                    continue
                face_track_embeddings[track_id] = np.mean(face_track_embeddings[track_id], axis=0)
            if cache:
                writeToPickleFile(face_track_embeddings, faceTrackEmbeddingsFilePath)
        return face_track_embeddings
        

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

    def run(self, fps=-1, resolution=None):
        framesObj = self.getVideoFrames(fps=fps, res=resolution)
        face_tracks = self.getFaceTracks(framesObj)
        face_track_embeddings =  self.getFaceTrackEmbeddings(framesObj, face_tracks)
        for track_id, track in face_tracks.items():
            face_track_embeddings[track_id] = {'track': track, 'embedding': face_track_embeddings[track_id]}
        ## sanity check face tracks  
        # self.visualizeFaceTracks(framesObj, face_tracks)
        return face_track_embeddings
    
if __name__ == "__main__":
    video_path = '/home/azureuser/cloudfiles/code/tsample3.mp4'

    cache_dir = os.path.join('/home/azureuser/cloudfiles/code/cache', os.path.basename(video_path)[:-4])
    os.makedirs(cache_dir, exist_ok=True)
    videoPreProcessor = VideoPreProcessor(video_path, cache_dir, verbose=True)
    videoPreProcessor.run()    