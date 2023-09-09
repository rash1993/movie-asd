import sys
sys.path.append('../voxceleb_trainer')
from torch.utils.data import DataLoader
from voxceleb_trainer.DatasetLoader import test_dataset_loader
from voxceleb_trainer.SpeakerNet import ModelTrainer, SpeakerNet, WrappedModel
import torch
import os
import glob
from tqdm import tqdm

class trainer(ModelTrainer):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):
        super().__init__(speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs)
    
    def evaluate(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=100, num_eval=10, **kwargs):
        self.__model__.eval()
        test_list = glob.glob(os.path.join(test_path, '*.wav'))
        test_list = [os.path.basename(d) for d in test_list]
        # print(len(test_list))
        feats = {}
        test_dataset = test_dataset_loader(test_list, test_path, num_eval=num_eval, **kwargs)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
            sampler=None
        )
        for idx, data in enumerate(tqdm(test_loader, desc='Extracting speaker features')):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                # print(inp1.shape, data[1][0])
                if inp1.shape[1] < 5000:
                    continue
                ref_feat = self.__model__(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat
        return feats

class SpeakerRecognition():
    def __init__(self):
        self.args = {'config': None,
                    'max_frames': 200,
                    'eval_frames': 0,
                    'batch_size': 200,
                    'max_seg_per_spk': 500,
                    'nDataLoaderThread': 5,
                    'augment': False,
                    'seed': 10,
                    'test_interval': 10,
                    'max_epoch': 500,
                    'trainfunc': 'softmaxproto',
                    'optimizer': 'adam',
                    'scheduler': 'steplr',
                    'lr': 0.001,
                    'lr_decay': 0.95,
                    'weight_decay': 0,
                    'hard_prob': 0.5,
                    'hard_rank': 10,
                    'margin': 0.1,
                    'scale': 30,
                    'nPerSpeaker': 1,
                    'nClasses': 5994,
                    'dcf_p_target': 0.05,
                    'dcf_c_miss': 1,
                    'dcf_c_fa': 1,
                    'initial_model': '../voxceleb_trainer/baseline_v2_ap.model',
                    'save_path': 'exps/exp1',
                    'train_list': 'data/train_list.txt',
                    'test_list': 'data/test_list.txt',
                    'train_path': 'data/voxceleb2',
                    'test_path': '/data/rash/active-speaker-detection/expts/Friends_s03e01/pyannote_vad_wavs',
                    'musan_path': 'data/musan_split',
                    'rir_path': 'data/RIRS_NOISES/simulated_rirs',
                    'n_mels': 64,
                    'log_input': True,
                    'model': 'ResNetSE34V2',
                    'encoder_type': 'ASP',
                    'nOut': 512,
                    'eval': True,
                    'port': '8888',
                    'distributed': False,
                    'mixedprec': False,
                    'gpu': 0}
        self.speaker_net = SpeakerNet(**self.args)
    
    def extractFeatures(self, wav_dir):
        self.args['gpu'] = 0
        self.args['test_path'] = wav_dir
        s = WrappedModel(self.speaker_net).cuda(self.args['gpu'])
        trainer_obj = trainer(s, **self.args)
        trainer_obj.loadParameters(self.args['initial_model'])
        feats = trainer_obj.evaluate(**self.args)
        return {name[:-4]: feat for name, feat in feats.items()}