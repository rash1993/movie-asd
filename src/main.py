'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-20 16:43:06
 * @modify date 2023-02-20 16:43:06
 * @desc [description]
 */'''

import argparse, sys, os
sys.path.append('../')
from preprocessor import Preprocessor
from ASD.ASD_framework import ASD

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--videoPath', type=str, required=True, help='video path')
    args.add_argument('--cacheDir', type=str, default='~/movie-asdache', help='path to store intermediate outputs')
    args.add_argument('--verbose', action='store_true', help='print the intermediate rocessing steps')
    args.add_argument('--partitionLength', type=int, default=-1, help='length of partition in number of speech segments')
    args = args.parse_args()
    videoName = os.path.basename(args.videoPath)[:-4]
    cacheDir = os.path.join(args.cacheDir, videoName)
    os.makedirs(cacheDir, exist_ok=True)

    preprocessor = Preprocessor(args.videoPath,\
                                cacheDir=cacheDir,\
                                verbose=args.verbose)
    preprocessor.prep()

    asdFramework = ASD(preprocessor.speechFaceTracks,\
                       preprocessor.videoPrep.faceTracks,\
                       preprocessor.videoPrep.faceTrackFeats,\
                       preprocessor.audioPrep.speechEmbeddings,\
                       cacheDir, guides=None, verbose=args.verbose, \
                       marginalFaceTracks=preprocessor.speechFaceTracksMarginal)
    if args.partitionLength == -1:
        prtitionLength = 'full'
    else:
        partitionLength = args.partitionLength
    asdFramework.run(partitionLen=partitionLength)
    asdFramework.visualizeASD(args.videoPath)
    asdFramework.visualizeDistanceMatrices()