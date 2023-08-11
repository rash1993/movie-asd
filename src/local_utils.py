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

def readVideoFrames(videoPath, res=(180, 360)):
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

    vid = cv2.VideoCapture(videoPath)
    FPS = vid.get(cv2.CAP_PROP_FPS)
    framesCount = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    if res:
        frameHeight, frameWidth = res
    else:
        frameHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frameWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    frames = []
    flag = True

    pbar = tqdm(total=framesCount, desc='reading video frames')
    while flag:
        flag, img = vid.read()
        if flag:
            if res:
                img = cv2.resize(img, (frameWidth, frameHeight))
            pbar.update(1)
    pbar.close()
    
    return {'frames':frames, 'height':frameHeight, 'width':frameWidth, 'fps':FPS}

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
