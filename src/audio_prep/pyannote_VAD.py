'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-04 19:14:17
 * @modify date 2023-02-04 19:14:17
 * @desc [description]
 */'''

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import numpy as np
from scipy.io import wavfile


class VoiceActivityDetector():
    def __init__(self, wavPath):
        self.wavPath = wavPath

    # def run(self):
    #     pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",\
    #         use_auth_token='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')
    #     output = pipeline(self.wavPath)
    #     output = [[d.start, d.end] for d in output.get_timeline()]
    #     output.sort(key = lambda x: x[0])
    #     output = np.array(output)
    #     return output

    def run(self):
        model = Model.from_pretrained("pyannote/segmentation",\
                use_auth_token='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')
        pipeline = VoiceActivityDetection(segmentation=model)
        HYPER_PARAMETERS = {"onset": 0.767,
                            "offset": 0.377,
                            "min_duration_on": 0.136,
                            "min_duration_off": 0.067}
        pipeline.instantiate(HYPER_PARAMETERS)
        vad = pipeline(self.wavPath)
        vad = [[d.start, d.end] for d in vad.get_timeline()]
        vad.sort(key=lambda x: x[0])
        vad = np.array(vad)
        return vad