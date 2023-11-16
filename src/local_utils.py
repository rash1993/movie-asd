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
            frames.append(img)
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

def npAngle(a, b, c):
    ba = a - b
    bc = c - b 
    
    cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def getEyeDistance(landmarks):
    return np.linalg.norm(landmarks[0] - landmarks[1])

def getFaceProfile(landmarks):
    angR = npAngle(landmarks[0], landmarks[1], landmarks[2]) # Calculate the right eye angle
    angL = npAngle(landmarks[1], landmarks[0], landmarks[2])
    if ((int(angR) in range(35, 57)) and (int(angL) in range(35, 58))):
        predLabel='Frontal'
    else: 
        if angR < angL:
            predLabel='Left Profile'
        else:
            predLabel='Right Profile'
    return predLabel

def inside(box1, box2):
  """Calculates the intersection over union of two bounding boxes.

  Args:
    box1: A list of four numbers [x1, y1, x2, y2] representing the coordinates of the first bounding box.
    box2: A list of four numbers [x1, y1, x2, y2] representing the coordinates of the second bounding box.

  Returns:
    The IoU of the two bounding boxes.
  """

  # Get the coordinates of the intersection rectangle.
  x1 = max(box1[0], box2[0])
  y1 = max(box1[1], box2[1])
  x2 = min(box1[2], box2[2])
  y2 = min(box1[3], box2[3])

  # If the intersection is empty, return 0.
  if x1 > x2 or y1 > y2:
    return 0

  # Calculate the area of the intersection rectangle.
  intersection_area = (x2 - x1) * (y2 - y1)
  box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
  return intersection_area / box1_area

def boxArea(box):
    return (box[2] - box[0])*(box[3] - box[1])

def plot_tracks(framesObj, tracks, color_dict={}):
    for trackId, track in tracks.items():
        if trackId in color_dict.keys():
            color = color_dict[trackId]
        else:
            color = list(np.random.random(size=3) * 256)
        for box in track:
            frameNo = int(round(box[0]*framesObj['fps']))
            x1 = int(round(box[1]*framesObj['width']))
            y1 = int(round(box[2]*framesObj['height']))
            x2 = int(round(box[3]*framesObj['width']))
            y2 = int(round(box[4]*framesObj['height']))
            cv2.rectangle(framesObj['frames'][frameNo], \
                          (x1, y1), (x2, y2),\
                          color = color,\
                          thickness = 2)
            cv2.putText(framesObj['frames'][frameNo], \
                        str(trackId), (x1+10, y1+10), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return framesObj

def writeVideo(frames, fps, width, height, path):
    video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), \
                                   fps, (int(width), int(height)))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()