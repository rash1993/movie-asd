'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-20 16:43:06
 * @modify date 2023-02-20 16:43:06
 * @desc [description]
 */'''

import argparse, sys, os, subprocess
sys.path.append('../')
from preprocessor import Preprocessor
from ASD.ASD_framework import ASD
from TalkNet.TalkNet_wrapper import TalkNetWrapper
import time

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--videoPath', type=str, required=True, help='video path')
    args.add_argument('--cacheDir', type=str, default='~/movie-asdache', help='path to store intermediate outputs')
    args.add_argument('--verbose', action='store_true', help='print the intermediate rocessing steps')
    args.add_argument('--partitionLength', type=int, default=-1, help='length of partition in number of speech segments')
    args.add_argument('--talknet', action='store_true', help='use the talknet as guides for CMIA')
    args = args.parse_args()
    videoName = os.path.basename(args.videoPath)[:-4]
    cacheDir = os.path.join(args.cacheDir, videoName)
    os.makedirs(cacheDir, exist_ok=True)

    st = time.time()
    #change the fps of the video to 25
    cache_video_path = os.path.join(cacheDir, os.path.basename(args.videoPath))
    if not os.path.isfile(cache_video_path):
        print('converting the video to 25fps')
        cmd = f"ffmpeg -y -i {args.videoPath} -filter:v fps=fps=25 {cache_video_path}"
        subprocess.call(cmd, shell=True, stdout=None)
    
    
    preprocessor = Preprocessor(cache_video_path,\
                                cacheDir=cacheDir,\
                                verbose=args.verbose,\
                                talknet_flag=args.talknet)
    preprocessor.prep()

    asdFramework = ASD(preprocessor.speechFaceTracks,\
                       preprocessor.videoPrep.faceTracks,\
                       preprocessor.videoPrep.faceTrackFeats,\
                       preprocessor.audioPrep.speechEmbeddings,\
                       cacheDir, guides=preprocessor.guides,\
                       verbose=args.verbose, \
                       marginalFaceTracks=preprocessor.speechFaceTracksMarginal)
    if args.partitionLength == -1:
        partitionLength = 'full'
    else:
        partitionLength = args.partitionLength
    asdFramework.run(partitionLen=partitionLength)
    asdFramework.visualizeASD(args.videoPath)
    asdFramework.visualizeDistanceMatrices()
    print(f'time elapsed: {time.time() - st}')