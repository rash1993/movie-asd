'''
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-02 13:44:00
 * @modify date 2023-02-02 13:44:00
 * @desc [description]
'''
import sys
sys.path.append('../../Pytorch_Retinaface')
# sys.path.append('../../sort')
import subprocess, csv, os, torch
import numpy as np
from sort.sort import Sort
from data import cfg_re50
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm
import pickle as pkl 
from tqdm import tqdm
from Pytorch_Retinaface.detect import load_model

class RetinaFaceWithSortTracker():
    def __init__(self, videoPath, framesObj):
        self.videoPath = videoPath
        self.videoName = os.path.basename(videoPath)[:-4]
        self.framesObj = framesObj
    
    def getShotBoundaries(self):
        shotsOutputFile = os.path.join('../../', f'{self.videoName}-Scenes.csv')
        scenedetectCmd = f'scenedetect --input {self.videoPath} --output ../../ detect-content list-scenes'
        subprocess.call(scenedetectCmd, shell=True, stdout=None)
        shots = list(csv.reader(open(shotsOutputFile, 'r'), delimiter = ','))
        del shots[0]
        del shots[0]
        self.shots = [[shot[0], float(shot[3]), float(shot[6])] for shot in shots]
        rmCmd = f'rm {shotsOutputFile}'
        subprocess.call(rmCmd, shell=True, stdout=False)
    
    def initRetinaFace(self):
        weights_file = '../../Pytorch_Retinaface/weights/Resnet50_Final.pth'
        cpu = False
        origin_size = True
        self.net_cfg = cfg_re50
        self.nms_threshold = 0.5
        self.confidence_threshold = 0.8
        
        self.net = RetinaFace(cfg=self.net_cfg, phase='test')
        self.net = load_model(self.net, weights_file, cpu) 
        self.net.eval()
        self.device = torch.device('cpu' if cpu else 'cuda')
        self.net = self.net.to(self.device)

    def getFaceTracksInShot(self, shot):  # sourcery skip: do-not-use-bare-except
        shotID, shotStartTime, shotEndTime = shot
        shotStartFrame = int(np.ceil(shotStartTime*self.framesObj['fps']))
        shortEndFrame = int(np.ceil(shotEndTime*self.framesObj['fps']))
        frames = self.framesObj['frames'][shotStartFrame : shortEndFrame]
        img_height, img_width, _ = frames[0].shape
        tracks = {}
        tracker = Sort()
        for ctr, img in enumerate(frames):
            img = np.float32(img)
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)
            loc, conf, landms = self.net(img)
            priorbox = PriorBox(self.net_cfg, image_size=(img_height, img_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.net_cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.net_cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                                img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1
            landms = landms.cpu().numpy()

            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]
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
                try:
                    tracks[track_label].append(box)
                except:
                    tracks[track_label] = [box]
        return tracks
         
   

    def run(self):
        tracks = {}
        self.getShotBoundaries()
        self.initRetinaFace()
        for shot in tqdm(self.shots, desc='extracting face tracks for each shot'):
            shotTracks = self.getFaceTracksInShot(shot)
            tracks |= shotTracks
        return tracks
        