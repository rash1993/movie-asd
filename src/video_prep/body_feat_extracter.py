import argparse
import os.path as osp
import os

import cv2
import numpy as np
from PIL import Image
import json
import sys
import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import *
from collections import OrderedDict
from datetime import datetime
from sklearn.preprocessing import  normalize

from video_prep.resnet import ResNet
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from tqdm import tqdm

class BodyCropsDataset(Dataset):
    def __init__(self, framesObj, bodyTracks, body_crop_h, body_crop_w) -> None:
        super().__init__()
        self.bodyTracks = bodyTracks
        self.framesObj = framesObj
        samples = []
        for trackId, track in self.bodyTracks.items():
            for i, box in enumerate(track):
                samples.append([f'{trackId}-{i}'] + box)
        
        normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((body_crop_h, body_crop_w)),
            normalizer])
        
        self.crops = []
        for sample in samples:
            boxId = sample[0]
            frameNo = int(round(sample[1]*self.framesObj['fps']))
            x1 = max(0, int(round(sample[2]*self.framesObj['width'])))
            y1 = max(0, int(round(sample[3]*self.framesObj['height'])))
            x2 = min(int(round(sample[4]*self.framesObj['width'])), self.framesObj['width'])
            y2 = min(int(round(sample[5]*self.framesObj['height'])), self.framesObj['height'])
            if (x1 >= x2) or (y1 >= y2):
                continue
            self.crops.append([boxId, frameNo, x1, y1, x2, y2])

    def __getitem__(self, index):
        crop = self.crops[index]
        boxId = crop[0]             
        frameNo = crop[1]
        x1, y1, x2, y2 = crop[2:]
        crop = self.framesObj['frames'][frameNo][y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = self.transform(crop)
        return boxId, crop

    def __len__(self):
        return len(self.crops)

class Extractor(object):
    def __init__(self, framesObj, bodyTracks, body_crop_w, body_crop_h):
        self.crop_w = body_crop_w
        self.crop_h = body_crop_h
        model = ResNet(50, pretrained=True, num_features=256, norm=True, dropout=0)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('../body_model.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model
        self.model.module.to('cuda')
        self.model.eval()
        dataset = BodyCropsDataset(framesObj, bodyTracks, body_crop_w, body_crop_h)
        sampler = SequentialSampler(dataset)
        self.dataLoader = DataLoader(dataset, batch_size=1024, sampler=sampler)

    def extract_features(self):
        outBoxIds = []
        outFeatures = []
        with torch.no_grad():
            for boxIds, boxes in tqdm(self.dataLoader):
                boxes = boxes.to('cuda')
                features = self.model(boxes)
                features = features.cpu().numpy()
                outBoxIds.extend(boxIds)
                outFeatures.extend(normalize(features))
        bodyTrackFeatures = {}
        for boxId, feature in zip(outBoxIds, outFeatures):
            trackId = boxId.split('-')[0]
            if trackId in bodyTrackFeatures.keys():
                bodyTrackFeatures[trackId].append(feature)
            else:
                bodyTrackFeatures[trackId] = [feature]
        
        for trackId, features in bodyTrackFeatures.items():
            bodyTrackFeatures[trackId] = np.mean(features, axis=0)
        return bodyTrackFeatures
