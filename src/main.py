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
from src.local_utils import visualize_asd, visualize_distance_matrix

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--videoPath', type=str, help='video path')
    args.add_argument('--cacheDir', type=str, help='path to store intermediate outputs')
    args.add_argument('--verbose', action='store_true', help='print the intermediate rocessing steps')
    args.add_argument('--fps', type=int, default=-1, help='frame rate for the video')
    args.add_argument('--resolution', type=tuple, default=(180, 360), help='resolution of the video')
    args = args.parse_args()
    videoName = os.path.basename(args.videoPath)[:-4]
    cacheDir = os.path.join(args.cacheDir, videoName)
    os.makedirs(cacheDir, exist_ok=True)

    speechFaceTracks, \
        faceTrackEmbeddings, \
            speechEmbeddings = Preprocessor(args.videoPath,\
                                            cacheDir=cacheDir,\
                                            verbose=args.verbose).prep(fps=args.fps,\
                                                                        resolution=args.resolution)

    asdFramework = ASD(cacheDir, verbose=args.verbose)
    asd_checkpoint = asdFramework.run(speechFaceTracks, faceTrackEmbeddings, speechEmbeddings)
    
    faceTracks = {track_id: track['track'] for track_id, track in faceTrackEmbeddings.items()}
    # visualize_asd(asd_checkpoint, cacheDir=cacheDir, videoPath=args.videoPath, faceTracks=faceTracks)
    visualize_distance_matrix(asd_checkpoint, cacheDir, speechEmbeddings, faceTrackEmbeddings)



    