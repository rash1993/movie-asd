'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-04 19:14:17
 * @modify date 2023-02-04 19:14:17
 * @desc [description]
 */'''

from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.utils.signal import Binarize
from pyannote.audio.utils.signal import Peak
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
        HYPER_PARAMETERS = {"onset": 0.6, "offset": 0.4, 
                  "min_duration_on": 0.0, "min_duration_off": 0.0}
        # HYPER_PARAMETERS = {"onset": 0.767,
        #                     "offset": 0.377,
        #                     "min_duration_on": 0.136,
        #                     "min_duration_off": 0.067}
        pipeline.instantiate(HYPER_PARAMETERS)
        vad = pipeline(self.wavPath)
        vad = [[d.start, d.end] for d in vad.get_timeline()]
        vad.sort(key=lambda x: x[0])
        vad = np.array(vad)
        return vad

class SpeakerHomogeneousSpeechSegmentation():
    def __init__(self, wavPath, auth='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'):
        self.wavPath = wavPath
        self.auth = auth
    
    def run(self):
        BATCH_AXIS = 0
        TIME_AXIS = 1
        SPEAKER_AXIS = 2
        to_vad = lambda o: np.max(o, axis=SPEAKER_AXIS, keepdims=True)
        vad = Inference("pyannote/segmentation", pre_aggregation_hook=to_vad,\
                        use_auth_token=self.auth)
        vad_prob = vad(self.wavPath)
        binarize = Binarize(onset=0.5)
        speech = binarize(vad_prob)
        to_scd = lambda probability: np.max(np.abs(np.diff(probability, n=1, axis=TIME_AXIS)), \
            axis=SPEAKER_AXIS, keepdims=True)
        scd = Inference("pyannote/segmentation", pre_aggregation_hook=to_scd, use_auth_token=self.auth)
        scd_prob = scd(self.wavPath)
        peak = Peak(alpha=0.05)
        scd = peak(scd_prob).crop(speech.get_timeline())
        scd = [[d.start, d.end] for d in scd]
        return scd