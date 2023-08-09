'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-02 16:43:13
 * @modify date 2023-02-02 16:43:13
 * @desc [description]
 */'''
 
import cv2, subprocess, os, csv
import pickle as pkl
from tqdm import tqdm 
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from scipy.io import wavfile

def timeCode2seconds(time_code):
    """method to convert time code to seconds

    Args:
        time_code (string): time code in the format hh:mm:ss

    Returns:
        seconds: time in seconds
    """
    time_code = [float(t) for t in time_code.split(':')]
    seconds = int(time_code[0]) * 3600 + int(time_code[1]) * 60 + float(time_code[2])
    return seconds

def cosine_dist(vector_i, vector_j):
    """method to calculate cosine distance between two vectors

    Args:
        vector_i (numpy array): vector i
        vector_j (numpy array): vector j

    Returns:
        cosine distance between vector i and vector j
    """
    return cdist(vector_i.reshape(1, -1), vector_j.reshape(1, -1), metric='cosine')[0, 0]

def readVideoFrames(video_path, fps=6, res=None):
    """method to read all the frames of a video file.
       Video must be in format supported by opencv.

    Args:
        videoPath (string): absolute path of the video file (.mp4 format is preferred)
        res (tuple, optional): resolution of the frames to be resized to (frameHeight, frameWidth). 
            Defaults to None.

    Returns:
        frames: list of the frames 
        frameHeight: height of the read frames.
        frameWidth: width of the read frames.
        fps: frame rate of the video file
    """

    vid = cv2.VideoCapture(video_path)
    original_framerate = vid.get(cv2.CAP_PROP_FPS)
    if fps < 0: 
        fps = original_framerate
    frame_skip = int(round(original_framerate / fps))
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if res:
        frameHeight, frameWidth = res
    else:
        frameHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frameWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    pbar = tqdm(total=total_frames, desc='reading frames')
    frame_counter = 0
    frames = []
    while vid.isOpened():
        ret, img = vid.read()
        if not ret:
            break
        if res:
            img = cv2.resize(img, (frameWidth, frameHeight))
        frames.append(img)
        frame_counter += 1
        for _ in range(frame_skip - 1):
            vid.grab()
            frame_counter += 1
        pbar.update(frame_skip)
    vid.release()
    return {'frames':frames, 'height':int(frameHeight), 'width':int(frameWidth), 'fps':fps}

def writeToPickleFile(obj, filePath):
    """method tp save an object in pickle format

    Args:
        obj (obj): any object (dictionary)
        filePath (string): path to store the object
    """
    with open(filePath, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    return

def shotDetect(videoPath, saveDir):
    videoName = os.path.basename(videoPath)[:-4]
    shotsOutputFile = os.path.join(f'{saveDir}', f'{videoName}-Scenes.csv')
    if not os.path.isfile(shotsOutputFile):
        scenedetectCmd = f'scenedetect --input {videoPath} \
                        --output {saveDir} detect-content list-scenes'
        subprocess.call(scenedetectCmd, shell=True, stdout=None)
    shots = list(csv.reader(open(shotsOutputFile, 'r'), delimiter = ','))
    del shots[0]
    del shots[0]
    shots = [[shot[0], float(shot[3]), float(shot[6])] for shot in shots]
    return shots

def split_face_tracks(face_tracks, scenes):
    '''
    split the faces tracks at the scene boundaries

    Args:
        face_tracks (list): list of face tracks
        scenes (list): list of scenes
    '''
    def get_start_end_time(face_track):
        return face_track[0][0], face_track[-1][0]
    
    face_tracks_keys = list(face_tracks.keys())
    face_tracks_keys.sort( key=lambda x: face_tracks[x][0][0])

    LEN_SCENES = len(scenes)
    while len(scenes):
        scene_idx = LEN_SCENES - len(scenes)
        curr_scene = scenes.pop(0)
        print(scene_idx)
        while len(face_tracks_keys):
            face_track_key = face_tracks_keys.pop(0)
            face_track = face_tracks[face_track_key]
            start_time, end_time = get_start_end_time(face_track)
            print(start_time, end_time, curr_scene)
            if start_time >= curr_scene[1]:
                break
            if end_time <= curr_scene[1]:
                # change the face track id by appending the scene id
                face_tracks[f'{scene_idx}_{face_track_key}'] = face_tracks[face_track_key]
                del face_tracks[face_track_key]
                continue
                # split the face track
            else:
                curr_scene_track = []
                next_scene_track = []
                for face in face_track:
                    if face[0] < curr_scene[1]:
                        curr_scene_track.append(face)
                    else:
                        next_scene_track.append(face)
                face_tracks[f'{scene_idx}_{face_track_key}'] = curr_scene_track
                face_tracks[face_track_key] = next_scene_track
                face_tracks_keys.insert(0, face_track_key)

    return face_tracks
            
def visualize_asd(asd_checkpoint, cacheDir, videoPath=None,framesObj=None, fps=-1, resolution=None, faceTracks=None):
    '''
    visualize the asd results
    '''
    if framesObj is None:
        if os.path.isfile(os.path.join(cacheDir, 'frames.pkl')):
            framesObj = pkl.load(open(os.path.join(cacheDir, 'frames.pkl'), 'rb'))
        else:
            framesObj = readVideoFrames(videoPath, fps=fps, res=resolution)
    frames = framesObj['frames']

    '''
    visualize the face tracks if provided
    '''
    if faceTracks is not None:
        for faceTrackId, face_track in faceTracks.items():
            for face in face_track:
                frame_no = int(round(face[0]*framesObj['fps']))
                x1 = int(face[1]*framesObj['width'])
                y1 = int(face[2]*framesObj['height'])
                x2 = int(face[3]*framesObj['width'])
                y2 = int(face[4]*framesObj['height'])
                cv2.rectangle(frames[frame_no], (x1, y1), (x2, y2), (255, 0, 0), 2)

    for segment_id, track in asd_checkpoint.items():
        for face in track['track']:
            # print(face)
            frame_no = int(round(face[0]*framesObj['fps']))
            x1 = int(face[1]*framesObj['width'])
            y1 = int(face[2]*framesObj['height'])
            x2 = int(face[3]*framesObj['width'])
            y2 = int(face[4]*framesObj['height'])
            cv2.rectangle(frames[frame_no], (x1, y1), (x2, y2), (0, 255, 0), 2)
    videoSavePath = os.path.join(cacheDir, 'asd_visualization.mp4')
    video_writer = cv2.VideoWriter(videoSavePath, cv2.VideoWriter_fourcc(*'mp4v'), \
                                   framesObj['fps'], (framesObj['width'], framesObj['height']))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
    print(f'asd video saved at {videoSavePath}')

        
def visualize_distance_matrix(asd_checkpoint, cacheDir, speechEmbeddings, faceTrackEmbeddings):
    distacnes_faces = np.zeros((len(asd_checkpoint.keys()), len(asd_checkpoint.keys())))
    distances_speech = np.zeros((len(asd_checkpoint.keys()), len(asd_checkpoint.keys())))
    for i, (segment_idi, tracki) in tqdm(enumerate(asd_checkpoint.items()), desc='computing distance matrices for visualization', total=len(asd_checkpoint.keys())):
        for j, (segment_idj, trackj) in enumerate(asd_checkpoint.items()):
            faceTrackEmbeddingi = faceTrackEmbeddings[tracki['track_id']]['embedding']
            faceTrackEmbeddingj = faceTrackEmbeddings[trackj['track_id']]['embedding']
            speechembeddingi = speechEmbeddings[segment_idi]['embedding']
            speechembeddingj = speechEmbeddings[segment_idj]['embedding']
            distacnes_faces[i, j] = cosine_dist(faceTrackEmbeddingi, faceTrackEmbeddingj)
            distances_speech[i,j] = cosine_dist(speechembeddingi, speechembeddingj)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(distacnes_faces)
    ax[1].imshow(distances_speech)
    plt.savefig(os.path.join(cacheDir, 'distance_matrix.png'), dpi=300)
        
def splitWav(wavPath, segments, cacheDir):
    """methods to split th audio file into smaller speaker homogeneous speech segments.
        Also ensures that each segment is greater the 0.4 sec in duration, as is required 
        by the speech recognition system.

    Args:
        segments (list): list of speaker homogeneous speech segments [segmentId, startTime, endTime]
        wavDir (string): directory path to save the wav files (Dafaults to None)
    
    Returns:
        None
        Stores the split wav files in cache dir
    """
    # check for minimum duration constraint
    rate, wavData = wavfile.read(wavPath)
    totalDur = len(wavData)/rate
    for i, segment in enumerate(segments):
        dur = segment[2] - segment[1]
        if dur < 0.4:
            center = (segment[1] + segment[2])/2
            st = max(0, center - 0.2)
            et = st + 0.4
            if et > totalDur:
                et = totalDur
                st = totalDur - 0.4
            segments[i] = [segment[0], st, et]
    wavDir = os.path.join(cacheDir, 'wavs')
    os.makedirs(wavDir, exist_ok=True)

    for segment in segments:
        segmentId, st, et = segment
        sf = int(st*rate)
        ef = int(et*rate)
        data_ = wavData[sf:ef]
        if len(data_) == 0:
            continue  
        savePath = os.path.join(wavDir, f'{segmentId}.wav')
        wavfile.write(savePath, rate, data_)
    return wavDir