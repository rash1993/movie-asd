import os
import csv
import numpy as np
import cv2
from tqdm import tqdm
from copy import deepcopy

class SceneExtractor():
    def __init__(self, video_path, base_dir='../expts'):
        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path)[:-4]
        self.base_dir = base_dir
    
    def extractScenes(self):
        self.scene_file = os.path.join(self.base_dir, self.video_name + '-Scenes.csv')
        if os.path.isfile(self.scene_file):
            print(f'scenes file already exists at: {self.scene_file}')
        else:
            cmd = f'scenedetect --input {self.video_path} --output {self.base_dir} detect-content \
                list-scenes'
            os.system(cmd)
            print(f'scenes saved at: {self.scene_file}')
        scenes = list(csv.reader(open(self.scene_file, 'r'), delimiter = ','))
        del scenes[0]
        del scenes[0]
        self.scenes = [[d[0], float(d[3]), float(d[6])] for d in scenes]
        
    def extractSceneFrames(self, redo=False):
        self.scene_dir = os.path.join(self.base_dir, self.video_name + '_scenes')
        os.makedirs(self.scene_dir, exist_ok=True)

        #check if the frame are already extracted, not a thorough check
        if not redo:
            status = []
            for scene in self.scenes:
                if os.path.isdir(os.path.join(self.scene_dir, scene[0])):
                    status.append(1)
                else:
                    status.append(0)
            if np.sum(status)/len(status) > 0.5:
                print('{} scenes already processed'.format(np.sum(status)/len(status)))
                return
        print(f'saving frames for {self.video_name} in {self.scene_dir}')
        
        vid = cv2.VideoCapture(self.video_path)
        fps = vid.get(5)
        # scenes = deepcopy(self.scenes)
        scenes = [[d[0], int(round(d[1]*fps)), int(round(d[2]*fps))] for d in self.scenes]
        total_number_scenes = len(scenes)
        pbar = tqdm(total = total_number_scenes)
        curr_scene = scenes.pop(0)
        
        scene_counter = 1
        os.makedirs(os.path.join(self.scene_dir, curr_scene[0]), exist_ok=True)
        flag = True
        frame_counter = 0
        while flag:
            
            flag, img = vid.read()
            if flag:
                frame_counter += 1
                if frame_counter >= curr_scene[2]:
                    try:
                        curr_scene = scenes.pop(0)
                        # pbar.update(total_number_scenes - len(scenes))
                        scene_counter += 1
                        # print(curr_scene[0])
                        os.makedirs(os.path.join(self.scene_dir, curr_scene[0]), exist_ok=True)
                        pbar.update(1)
                        cv2.imwrite(os.path.join(self.scene_dir, curr_scene[0],\
                            f'{frame_counter}.png'), img)
                    except:
                        break
                else:
                    cv2.imwrite(os.path.join(self.scene_dir, curr_scene[0], f'{frame_counter}.png'), img)        
        pbar.close()

    def run(self):
        self.extractScenes()
        self.extractSceneFrames()
    