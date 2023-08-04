'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-04 17:33:35
 * @modify date 2023-02-04 17:33:35
 * @desc [description]
 */'''
import os, subprocess, sys, torch
from src.audio_prep.vad import VoiceActivityDetector as VAD
from src.local_utils import writeToPickleFile, timeCode2seconds
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from scenedetect import detect, AdaptiveDetector, ContentDetector
from scipy.io import wavfile
from tqdm import tqdm
import pickle as pkl 
import numpy as np

class AudioPreProcessor():
    def __init__(self, videoPath, cacheDir, verbose=False):
        self.videoPath = videoPath
        self.cacheDir = cacheDir
        self.videoName = os.path.basename(videoPath)[:-4]
        self.verbose = verbose
    
    def getAudioWav(self, cache=True):
        wavPath = os.path.join(self.cacheDir, 'audio.wav')
        if os.path.isfile(wavPath) and cache:
            if self.verbose:
                print('using wav file from cache')
        else:
            if self.verbose:
                print(f'extracting the wav file and saving at: {wavPath}')
            wavCmd = f'ffmpeg -y -nostdin -loglevel error -y -i {self.videoPath} \
                -ar 16k -ac 1 {wavPath}'
            subprocess.call(wavCmd, shell=True, stdout=False)
        return wavPath
    
    def getVoiceAvtivity(self, wavPath, cache=True):
        self.vadPath = os.path.join(self.cacheDir, 'vad.pkl')
        if os.path.isfile(self.vadPath) and cache:
            if self.verbose:
                print('using vad from cache')
            vad = pkl.load(open(self.vadPath, 'rb'))
        else:
            if self.verbose:
                print(f'extracting the vad and saving at: {self.vadPath}')
            vad = VAD(wavPath).run()
            writeToPickleFile(vad, self.vadPath)
        return vad       

    def getSpeakerHomogeneousSegments(self, vad, maxth=1.0):
        """method to generate speaker homogeneous speech segments. Using a proxy that segments 
        the VAD speech segments at shot boundaries and make sure their max duration is maxth (1.0s)

        Args:
            maxth (float, optional): Maximum duration of each output segment. Defaults to 1.0.

        Returns:
            segments (list):[segment_id, start_time, end_time]
        """
        shots_file = os.path.join(self.cacheDir, 'scenes.csv')
        if os.path.isfile(shots_file):
            shots = np.loadtxt(shots_file, delimiter=',', dtype=str)
            shots = [[float(shot[0]), float(shot[1])] for shot in shots]
        else:
            shots = detect(self.videoPath, ContentDetector())
            shots  = [[timeCode2seconds(shot[0].get_timecode()),  \
                       timeCode2seconds(shot[1].get_timecode())] for shot in shots]
        segments = {}
        for shotId, shot in enumerate(shots):
            shot_st, shot_et = shot
            counter = 0
            for segment in vad:
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
        
        return([[name] + value for name, value in segments.items()])

    def extractSpeechEmbeddings(self, wavPath, segments, cache=True) -> dict:
        """method to extract speaker embeddings from the speaker homogeneous speech segments

        Args:
            segments (list): list of speaker homogeneous speech segments [segmentId, startTime, endTime]
            wavDir (string): directory path to save the wav files (Dafaults to None)
        
        Returns:
            SpeechEmbeddings (dict): {segmentId: speakerEmbedding}
            Stores the speaker embeddings in cache dir
        """
        # check for minimum duration constraint
        speechEmbeddingsPath = os.path.join(self.cacheDir, 'speechEmbeddings.pkl')
        if os.path.isfile(speechEmbeddingsPath) and cache:
            if self.verbose:
                print('using speech embeddings from cache')
            speechEmbeddings = pkl.load(open(speechEmbeddingsPath, 'rb'))
        else:
            if self.verbose:
                print(f'extracting the speech embeddings and saving at: {speechEmbeddingsPath}')
            rate, wavData = wavfile.read(wavPath)
            totalDur = len(wavData)/rate
            for i, segment in enumerate(segments):
                dur = segment[2] - segment[1]
                if dur < 0.4:
                    center = (segment[1] + segment[2])/2
                    st = max(0, center - 0.2)
                    et = st + 0.4
                    if et > totalDur-0.1:
                        # margin of 0.1 sec
                        et = totalDur -0.1
                        st = et - 0.4
                    segments[i] = [segment[0], st, et]
            # adding a margin of 0.1 sec if the et > totalDur
            segments = [[segment[0], segment[1], min(segment[2], totalDur-0.1)] for segment in segments]
            
            speechEmbeddings = {}
            audio = Audio(sample_rate=16000, mono="downmix")
            model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",device=torch.device("cuda"))
            for segment in tqdm(segments, desc='extracting speech embeddings'):
                segmentId, st, et = segment
                segment = Segment(st, et)
                waveform, sample_rate = audio.crop(wavPath, segment)
                speechEmbeddings[segmentId] = model(waveform[None])
            writeToPickleFile(speechEmbeddings, speechEmbeddingsPath)
        return speechEmbeddings
        
    def run(self):
        wavPath = self.getAudioWav()
        print(wavPath)
        vad = self.getVoiceAvtivity(wavPath)
        speakerHomoSegments = self.getSpeakerHomogeneousSegments(vad)
        speechEmbeddings = self.extractSpeechEmbeddings(wavPath, speakerHomoSegments)
        for segment in speakerHomoSegments:
            speechEmbeddings[segment[0]] = {'segment': segment[1:], 'embedding': speechEmbeddings[segment[0]]}
        return speechEmbeddings

if __name__ == '__main__':
    video_path = '/home/azureuser/cloudfiles/code/tsample3.mp4'

    cache_dir = os.path.join('/home/azureuser/cloudfiles/code/cache', os.path.basename(video_path)[:-4])
    os.makedirs(cache_dir, exist_ok=True)
    audioPreProcessor = AudioPreProcessor(video_path, cache_dir, verbose=True)
    audioPreProcessor.run()