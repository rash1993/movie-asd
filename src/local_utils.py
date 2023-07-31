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
    pbar = tqdm(total=total_frames)
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
            


        
    
        
