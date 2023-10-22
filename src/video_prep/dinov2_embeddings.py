'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-10-22 10:31:44
 * @modify date 2023-10-22 10:31:44
 * @desc [file to generate huamn boddy (attire) embeddings using dino_v2]
 */'''

import torch, os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from local_utils import boxArea


class BodyTracksDataset(Dataset):
    def __init__(self, frameObj, bodyTracks, maxSamplesPerTrack=4) -> None:

        super().__init__()
        self.framesObj = frameObj
        self.bodyTracks = bodyTracks
        self.transforms = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
        # select maxSamplesPerTrack and make a list with id
        self.samples = []
        for trackId, bodyTrack in bodyTracks.items():
            bodyTrack.sort(key=lambda x: boxArea(x), reverse=True)
            bodyTrack = bodyTrack[:maxSamplesPerTrack]
            for box in bodyTrack:
                frameNum = int(round(box[0]*self.framesObj['fps']))
                x1 = int(np.max((0, round(box[1]*self.framesObj['width']))))  # type: ignore                
                y1 = int(np.max((0, round(box[2]*self.framesObj['height'])))) #type: ignore
                x2 = int(np.min((self.framesObj['width'], \
                                    round(box[3]*self.framesObj['width'])))) #type: ignore
                y2 = int(np.min((self.framesObj['height'], \
                                    round(box[4]*self.framesObj['height'])))) #type: ignore
                if (x1 >= x2) or (y1 >= y2):
                    continue
                else:
                    self.samples.append([trackId, frameNum, x1, y1, x2, y2])
            
    def __getitem__(self, index):
        trackId, frameNum, x1, y1, x2, y2 = self.samples[index]
        crop = self.framesObj['frames'][frameNum][y1:y2, x1:x2]
        crop = self.transforms(crop)
        return trackId, crop
    
    def __len__(self):
        return len(self.samples)
        

class DinoV2Embeddings():
    def __init__(self):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.to('cuda')
        # self.transforms = T.Compose([T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
    
    def extractFeatures(self, dataLoader):
        outputs = {}
        with torch.no_grad():
            for trackIds, crops in tqdm(dataLoader, desc='DinoV2 inferences'):
                # crops.to('cuda')
                # crops = self.transforms(crops)
                embeddings = self.model(crops.to('cuda'))
                embeddings = embeddings.cpu().numpy()
                for Id, embedd in zip(trackIds, embeddings):
                    if Id in outputs.keys():
                        outputs[Id].append(embedd)
                    else:
                        outputs[Id] = [embedd]
        return outputs

    def run(self, framesObj, bodyTracks):
        dataset = BodyTracksDataset(framesObj, bodyTracks)
        sampler = SequentialSampler(dataset)
        dataLoader = DataLoader(dataset, batch_size=512, sampler=sampler)
        outputs = self.extractFeatures(dataLoader)
        for trackId, embeddings in outputs.items():
            if len(embeddings):
                outputs[trackId] = np.mean(embeddings, axis=0)
            else:
                outputs[trackId].pop()
        return outputs
