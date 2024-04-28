import argparse
import json
import gc
import os
import time
from datetime import datetime
import torch
from nnAudio import features as nnAudioFeatures
import numpy as np
from datasets import load_dataset, load_from_disk
from collections import defaultdict
from audiotools import AudioSignal
from SoundCodec.base_codec.general import pad_arrays_to_match
from metrics import get_metrics
import psutil
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

class CQTLoss(torch.nn.Module):
    def __init__(self, n_bins=84, sr=32000, freq=50):
        super().__init__()
        self.epsilon=1e-10
        # Getting Mel Spectrogram on the fly
        self.spec_layer = nnAudioFeatures.cqt.CQT(sr=sr, hop_length=sr//freq, fmin=32.7, 
                                           fmax=None, n_bins=n_bins, bins_per_octave=n_bins//7, 
                                           filter_scale=1, norm=1, window='hann', center=True, 
                                           pad_mode='constant', trainable=False, 
                                           output_format='Magnitude', verbose=True)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, y):
        '''
        take input from transformer hidden states: [batch, len_seq, channel]
        output: [batch, len_seq, n_bins]
        '''
        cqt_target = torch.transpose(self.spec_layer(x), -1, -2)
        cqt_preds = torch.transpose(self.spec_layer(y), -1, -2)
        return self.criterion(cqt_target, cqt_preds).item()

cqt = CQTLoss()
c = load_from_disk('datasets/audio-2m-valid_synth')
c = c.filter(lambda x: x['duration']<=60, num_proc=16)
models = [key for key in c.keys() if key != "original"]
result_data = {}
for model in models:
    args_list = [(original_iter, model_iter, 60) for original_iter, model_iter in
                     tqdm(zip(c['original'], c[model]), total=len(c[model]), desc='loading audio arrays')]
    metrics_results = []
    for args in tqdm(args_list):
        original, model_, max_duration = args
        original_arrays, resynth_array = pad_arrays_to_match(original['audio']['array'], model_['audio']['array'])
        sampling_rate = original['audio']['sampling_rate']
        original_signal = AudioSignal(original_arrays, sampling_rate)
        if original_signal.duration > max_duration:
            continue
        model_signal = AudioSignal(resynth_array, sampling_rate)
        # metrics = get_metrics(original_signal, model_signal)
        metrics = cqt(torch.FloatTensor(original_arrays), torch.FloatTensor(resynth_array))
        metrics_results.append(metrics)
    print(f'{model} cqt loss: {np.mean(metrics_results)}')