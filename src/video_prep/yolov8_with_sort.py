'''
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-10-19 20:51:17
 * @modify date 2023-10-19 20:51:17
 * @desc [description]
'''

import os
import numpy as np
from tqdm import tqdm
from ultralyticsplus import YOLO
from sort.sort import Sort
from local_utils import shotDetect
import torch

def split_lines(lines, batch_size):
    splits = []
    for i in range(0, len(lines), batch_size):
        splits.append(lines[i: i+ batch_size])
    return splits


class YoloWithSortTracker():
    def __init__(self, videoPath, cacheDir, frameObj):
        self.videoPath = videoPath
        self.videoName = os.path.basename(videoPath)[:-4]
        self.framesObj = frameObj
        self.cacheDir = cacheDir
    
    def initYolov8(self):
        model = YOLO('ultralyticsplus/yolov8s')
        model.overrides['conf'] = 0.25  # NMS confidence threshold
        model.overrides['iou'] = 0.45  # NMS IoU threshold
        model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        model.overrides['max_det'] = 1000
        self.model = model   
    
    def getBodyTrackInShot(self, shot):
        shotID, shotStartTime, shotEndTime = shot
        shotStartFrame = int(np.ceil(shotStartTime*self.framesObj['fps']))
        shortEndFrame = int(np.ceil(shotEndTime*self.framesObj['fps']))
        frames = self.framesObj['frames'][shotStartFrame : shortEndFrame]
        img_height, img_width, _ = frames[0].shape
        tracks = {}
        tracker = Sort()

        frames_splits = split_lines(frames, batch_size=512)
        detections = []
        for frames_split in frames_splits:
            detections.extend(self.model.predict(frames_split, verbose=False))

        
        for ctr, frame_dets in enumerate(detections):
            dets = []
            for box in frame_dets:
                box = box.boxes.cpu()
                xyxy = box.xyxy[0].numpy()
                cls = box.cls[0].numpy()
                conf = box.conf[0].numpy()
                if int(cls) != 0:
                    continue
                dets.append(xyxy.tolist() + [conf])
            dets = np.array(dets).reshape(-1, 5)
            
            track_bbs_ids = tracker.update(dets)

            for j in range(track_bbs_ids.shape[0]):
                ele = track_bbs_ids[j, :]
                x = int(ele[0])/self.framesObj['width']
                y = int(ele[1])/self.framesObj['height']
                x2 = int(ele[2])/self.framesObj['width']
                y2 = int(ele[3])/self.framesObj['height']
                track_label = f'{shotID}_{int(ele[4])}'
                time = ctr/self.framesObj['fps'] + shotStartTime
                box = [time, x, y, x2, y2]
                if track_label in tracks.keys():
                    tracks[track_label].append(box)
                else:
                    tracks[track_label] = [box]
        return tracks

    def run(self):
        tracks = {}
        self.shots = shotDetect(self.videoPath, self.cacheDir)
        self.initYolov8()
        for shot in tqdm(self.shots, desc='extracting body tracks for each shot'):
            shotTracks = self.getBodyTrackInShot(shot)
            tracks.update(shotTracks)
        torch.cuda.empty_cache()
        return tracks            