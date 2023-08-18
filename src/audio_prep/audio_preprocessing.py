'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-04 17:33:35
 * @modify date 2023-02-04 17:33:35
 * @desc [description]
 */'''
import os, subprocess, sys
from audio_prep.pyannote_VAD import VoiceActivityDetector as pyannoteVAD
from audio_prep.pyannote_VAD import SpeakerHomogeneousSpeechSegmentation as pyannoteSCD
from local_utils import shotDetect
from scipy.io import wavfile
from audio_prep.resnet_clovaAI import SpeakerRecognition
import pickle as pkl 
from local_utils import writeToPickleFile
import numpy as np

class AudioPreProcessor():
    def __init__(self, videoPath, cacheDir, verbose=False):
        self.videoPath = videoPath
        self.cacheDir = cacheDir
        self.videoName = os.path.basename(videoPath)[:-4]
        self.verbose = verbose
    
    def getAudioWav(self):
        self.wavPath = os.path.join(self.cacheDir, 'audio.wav')
        if os.path.isfile(self.wavPath):
            if self.verbose:
                print('using wav file from cache')
        else:
            if self.verbose:
                print(f'extracting the wav file and saving at: {self.wavPath}')
            wavCmd = f'ffmpeg -y -nostdin -loglevel error -y -i {self.videoPath} \
                -ar 16k -ac 1 {self.wavPath}'
            subprocess.call(wavCmd, shell=True, stdout=False)
    
    def getVoiceAvtivity(self, VAD='pyannote'):
        vadFile = os.path.join(self.cacheDir, 'vad.pkl')
        if os.path.isfile(vadFile):
            if self.verbose:
                print('reading VAD from cache')
            self.vad = pkl.load(open(vadFile, 'rb'))
        else:
            if self.verbose:
                print('computing VAD')
            self.vad = pyannoteVAD(self.wavPath).run()
            writeToPickleFile(self.vad, vadFile)

    def getSpeakerHomogeneousSegmentsPyannote(self):
        scd = pyannoteSCD(self.wavPath).run()
        self.speakerHomoSegments = [[f'{i}'] + segment_ for i, segment_ in enumerate(scd)]


    def getSpeakerHomogeneousSegments(self, maxth=1.0):
        """method to generate speaker homogeneous speech segments. Using a proxy that segments 
        the VAD speech segments at shot boundaries and make sure their max duration is maxth (1.0s)

        Args:
            maxth (float, optional): Maximum duration of each output segment. Defaults to 1.0.

        Returns:
            segments (list):[segment_id, start_time, end_time]
        """
        shots = shotDetect(self.videoPath, self.cacheDir)
        segments = {}
        for shot in shots:
            shotId, shot_st, shot_et = shot
            counter = 0
            for segment in self.vad:
                if segment[0] > shot_et:
                    break
                overlap = min(shot_et, segment[1]) - max(shot_st, segment[0])
                if overlap > 0:
                    st_ = max(shot_st, segment[0])
                    et_ = min(shot_et, segment[1])
                    flag = True
                    while flag:
                        segment_id = f'{shotId}_{counter}'
                        segments[segment_id] = [st_, min(et_, st_ + maxth)]
                        st_ = st_ + maxth
                        counter += 1
                        if st_ >= et_:
                            flag = False
        
        self.speakerHomoSegments = [[name] + value for name, value in segments.items()]

    def splitWav(self, segments, wavDir=None):
        """methods to split th audio file into smaller speaker homogeneous speech segments.
           Also ensures that each segment is greater the 0.4 sec in duration, as is required 
           by the speech recognition system.

        Args:
            segments (list): list of speaker homogeneous speech segments [segmentId, startTime, endTime]
            wavDir (string): directory path to save the wav files (Dafaults to None)
        
        Returns:
            None
            Stores the split wav files in cache dir
        """
        # check for minimum duration constraint
        rate, wavData = wavfile.read(self.wavPath)
        totalDur = len(wavData)/rate
        for i, segment in enumerate(segments):
            dur = segment[2] - segment[1]
            if dur < 0.4:
                center = (segment[1] + segment[2])/2
                st = max(0, center - 0.2)
                et = st + 0.4
                if et > totalDur:
                    et = totalDur
                    st = totalDur - 0.4
                segments[i] = [segment[0], st, et]
        if wavDir is None:
            wavDir = os.path.join(self.cacheDir, 'wavs')
        os.makedirs(wavDir, exist_ok=True)

        for segment in segments:
            segmentId, st, et = segment
            sf = int(st*rate)
            ef = int(et*rate)
            data_ = wavData[sf:ef]
            if len(data_) == 0:
                continue  
            savePath = os.path.join(wavDir, f'{segmentId}.wav')
            wavfile.write(savePath, rate, data_)
        return wavDir

    def extractSpeechEmbeddings(self):
        speechEmbeddingsFile = os.path.join(self.cacheDir, 'speechEmbeddings.pkl')
        if os.path.isfile(speechEmbeddingsFile):
            if self.verbose:
                print('reading speech embeddings from cache')
            self.speechEmbeddings = pkl.load(open(speechEmbeddingsFile, 'rb'))
        else:
            wavDir = self.splitWav(self.speakerHomoSegments)
            self.speechEmbeddings = SpeakerRecognition().extractFeatures(wavDir)
            writeToPickleFile(self.speechEmbeddings, speechEmbeddingsFile)
            rmCmd = f'rm -r {wavDir}'
            subprocess.call(rmCmd, shell=True, stdout=False)
    
    def run(self):
        self.getAudioWav()
        self.getVoiceAvtivity()
        self.getSpeakerHomogeneousSegments()
        # self.getSpeakerHomogeneousSegmentsPyannote()
        self.extractSpeechEmbeddings()