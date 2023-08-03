import torch
import pickle
import numpy as np
import os

import audonnx

from preprocess import load_pickle_file
from dataset import VCDataset
from model import Generator, Discriminator

vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')


def mel_decoder(vocoder, mel, mel_mean, mel_std):
    denorm_converted = mel * mel_std + mel_mean
    rev = vocoder.inverse(denorm_converted)
    return rev


torch.cuda.set_device(0)
device = "cuda"
g_lr = 2e-4
d_lr = 1e-4
# we can test 0.2 for two steps modify
emotion_modify_level = 0.4
fs = 16000

dataset_path = os.path.join(".", "dataset")
preprocessed_path = os.path.join(".", "preprocess")

dataset = VCDataset(load_pickle_file(os.path.join(preprocessed_path, "normalized.pickle")))
dataset_norm_status = np.load(os.path.join(preprocessed_path, "norm_status.npz"))
mean = dataset_norm_status['mean']
std = dataset_norm_status['std']

train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False)

g = Generator().to(device=device)
g2 = Generator().to(device=device)
d = Discriminator().to(device=device)
emotion_discriminator = audonnx.load(".")

g_params = list(g.parameters()) + list(g2.parameters())
d_params = list(d.parameters())

g_optimizer = torch.optim.Adam(g_params, lr=g_lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(d_params, lr=d_lr, betas=(0.5, 0.999))

i = 0
for real_data in train_dataloader:
    real_data = real_data.to(device, dtype=torch.float)

    fake_data = g(real_data)
    cycle_data = g2(fake_data)

    d_real_source = d(real_data)
    d_fake_source = d(fake_data)
    d_cycle_source = d(cycle_data)

    real_wav = mel_decoder(vocoder, real_data.detach().cpu(), mean, std)
    fake_wav = mel_decoder(vocoder, fake_data.detach().cpu(), mean, std)
    cycle_wav = mel_decoder(vocoder, cycle_data.detach().cpu(), mean, std)

    emotion_source_real = torch.tensor(emotion_discriminator(real_wav, fs)['logits'][0][0], device=device, dtype=torch.float)
    emotion_source_fake = torch.tensor(emotion_discriminator(fake_wav, fs)['logits'][0][0], device=device, dtype=torch.float)
    emotion_source_cycle = torch.tensor(emotion_discriminator(cycle_wav, fs)['logits'][0][0], device=device, dtype=torch.float)

    #g loss
    fake_loss = torch.mean(torch.abs(1 - d_fake_source))
    cycle_loss = torch.mean(torch.abs(real_data - cycle_data))

    #emotion g loss
    fake_emotion_loss = torch.abs(emotion_source_fake - (emotion_source_real + emotion_modify_level)) * 10
    cycle_emotion_loss = torch.abs(emotion_source_real - emotion_source_cycle) * 5

    g_loss = fake_loss + cycle_loss + fake_emotion_loss + cycle_emotion_loss

    #d loss
    d_real_loss = torch.mean(torch.abs(1 - d_real_source))
    d_fake_loss = torch.mean(torch.abs(0 - d_fake_source))
    d_cycle_loss = torch.mean(torch.abs(1 - d_cycle_source))

    d_loss = d_real_loss + d_fake_loss + d_cycle_loss

    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    g_loss.backward()
    d_loss.backward()

    g_optimizer.step()
    d_optimizer.step()
