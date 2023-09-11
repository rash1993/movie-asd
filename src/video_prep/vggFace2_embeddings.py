'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-04 09:30:07
 * @modify date 2023-02-04 09:30:07
 * @desc [description]
 */
'''
import numpy as np
from tqdm import tqdm
from Keras_VGGFace2_ResNet50.src.wrapper import initialize_model, image_encoding
from local_utils import getEyeDistance
from collections import Counter

class VggFace2Embeddings():
    def __init__(self, framesObj, faceTracks):
        self.framesObj = framesObj
        self.faceTracks = faceTracks
    
    def selBestFaces(self, faceTrack, N=4):
        # select the top N faces with the maximum distances between the eyes\
        # as a proxy for the frontal faces. 
        faceTrack.sort(key=lambda x: getEyeDistance(np.array(x[-1]).reshape((-1, 2))), reverse=True)
        return faceTrack[:N]

    def extractCrops(self, faceTracksList):
        # extracting face crops
        crops = []
        cropIDs = []
        for faceTrack in tqdm(faceTracksList, desc='extracting image crops for face tracks'):
            bestFaces = self.selBestFaces(self.faceTracks[faceTrack])
            for box in bestFaces:
                frameNum = int(round(box[0]*self.framesObj['fps']))
                x1 = int(np.max((0, round(box[1]*self.framesObj['width']))))  # type: ignore                
                y1 = int(np.max((0, round(box[2]*self.framesObj['height'])))) #type: ignore
                x2 = int(np.min((self.framesObj['width'], round(box[3]*self.framesObj['width'])))) #type: ignore
                y2 = int(np.min((self.framesObj['height'], round(box[4]*self.framesObj['height'])))) #type: ignore
                if (x1 >= x2) or (y1 >= y2):
                    continue
                else:
                    crop = self.framesObj['frames'][frameNum][y1:y2, x1:x2]
                crops.append(crop)
                cropIDs.append(faceTrack)
        
        return crops, cropIDs
    
    def extractEmbeddings(self, faceTracksList):
        crops, cropIDs = self.extractCrops(faceTracksList)
        # print(Counter(cropIDs))
        vggFace2Model = initialize_model()
        cropFeats = image_encoding(vggFace2Model, crops)
        faceTrackFeats = {}
        for faceTrack in tqdm(self.faceTracks.keys(), desc='extracting face track features'):
            faceTrackCrops = [_cropFeat for _cropFeat, _cropID in zip(cropFeats, cropIDs) \
                if _cropID == faceTrack]
            if len(faceTrackCrops):
                faceTrackFeats[faceTrack] = np.mean(faceTrackCrops, axis=0)
        return faceTrackFeats