from deepface import DeepFace

class FaceDetector:
    def __init__(self):
        self.backend = 'retinaface'

    def run(self, frames):
        faces = []
        for frame in frames:
            faces = DeepFace.extract_faces(frame['frame'], backend=self.backend)

