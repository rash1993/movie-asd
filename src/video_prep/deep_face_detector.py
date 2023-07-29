from deepface import DeepFace
from collections import defaultdict
from tqdm import tqdm

def dd():
    return []
    
class FaceDetector:
    def __init__(self, backend='retinaface'):
        self.backend = backend

    def run(self, frames):
        x_scale = frames[0].shape[1]
        y_scale = frames[0].shape[0]
        faces_out = []
        for frame in tqdm(frames, desc='extracting faces'):
            frame_faces = DeepFace.extract_faces(img_path = frame, \
                                                 detector_backend=self.backend)
            boxes = []
            for bbox in frame_faces:
                bbox = bbox['facial_area'] + [bbox['confidence']]
                bbox = list(bbox.values())
                # of format (x1, y1, x2, y2)
                bbox = [bbox[0]/x_scale, \
                        bbox[1]/y_scale, \
                        (bbox[0] + bbox[2])/x_scale, \
                        (bbox[1] + bbox[3])/y_scale, \
                        bbox[4]]
                boxes.append(bbox)
            faces_out.append(boxes)
        return faces_out

