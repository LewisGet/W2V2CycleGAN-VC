"""
MaskCycleGAN-VC models as described in https://arxiv.org/pdf/2102.12841.pdf
this code copy from https://github.com/GANtastic3/MaskCycleGAN-VC
"""

import os
import pickle
import glob
import numpy as np
import librosa
import torch

SAMPLING_RATE = 16000  # Fixed sampling rate


def normalize_mel(wavspath):
    wav_files = glob.glob(wavspath, recursive=True)  # source_path
    vocoder = torch.hub.load('LewisGet/melgan-neurips', 'load_melgan')

    mel_list = list()
    for wavpath in wav_files:
        wav, _ = librosa.load(wavpath, sr=SAMPLING_RATE, mono=True)
        mel = vocoder(torch.tensor([wav]))
        mel_list.append(mel.cpu().detach().numpy()[0])

    mel_concatenated = np.concatenate(mel_list, axis=1)
    mel_mean = np.mean(mel_concatenated, axis=1, keepdims=True)
    mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9

    mel_normalized = list()
    for mel in mel_list:
        if mel.shape[-1] < 64:
            continue

        app = (mel - mel_mean) / mel_std
        mel_normalized.append(app)

    return mel_normalized, mel_mean, mel_std


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def preprocess_dataset(data_path, preprocess_folder='preprocess'):
    mel_normalized, mel_mean, mel_std = normalize_mel(data_path)

    if not os.path.exists(os.path.join(".", preprocess_folder)):
        os.makedirs(os.path.join(".", preprocess_folder))

    np.savez(os.path.join(preprocess_folder, "norm_status.npz"),
             mean=mel_mean,
             std=mel_std)

    save_pickle(variable=mel_normalized,
                fileName=os.path.join(".", preprocess_folder, "normalized.pickle"))


if __name__ == '__main__':
    preprocess_dataset(os.path.join(".", "train_dataset", "*.wav"))
