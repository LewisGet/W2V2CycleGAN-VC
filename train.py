import torch
import torchaudio
import numpy as np
import os
from config import *

from preprocess import load_pickle_file
from dataset import VCDataset
from model import Generator, Discriminator, EmotionModel
from transformers import Wav2Vec2Processor

from torch.utils.tensorboard import SummaryWriter


vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

emotion_model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(emotion_model_name)
emotion_discriminator = EmotionModel.from_pretrained(emotion_model_name)


def mel_decoder(vocoder, mel, mel_mean, mel_std):
    denorm_converted = mel * mel_std + mel_mean
    rev = vocoder.inverse(denorm_converted)
    return rev


dataset = VCDataset(load_pickle_file(os.path.join(preprocessed_path, "normalized.pickle")))
dataset_norm_status = np.load(os.path.join(preprocessed_path, "norm_status.npz"))
mean = dataset_norm_status['mean']
std = dataset_norm_status['std']

train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False)

g = Generator().to(device=device)
g2 = Generator().to(device=device)
d = Discriminator().to(device=device)
emotion_discriminator = emotion_discriminator.to(device=device)

g_params = list(g.parameters()) + list(g2.parameters())
d_params = list(d.parameters())

g_optimizer = torch.optim.Adam(g_params, lr=g_lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(d_params, lr=d_lr, betas=(0.5, 0.999))

if loading_model > 0:
    loading_names = ["g", "g2", "d", "g_optimizer", "d_optimizer"]

    for variable_name in loading_names:
        vars()[variable_name].load_state_dict(torch.load(os.path.join(model_path, "%s-%d.ckpt" % (variable_name, loading_model))))

writer = SummaryWriter()

for i in range(steps):
    if i == 0:
        i = loading_model

    for real_data in train_dataloader:
        real_data = real_data.to(device, dtype=torch.float)

        fake_data = g(real_data)
        cycle_data = g2(fake_data)

        d_real_source = d(real_data)
        d_fake_source = d(fake_data)
        d_cycle_source = d(cycle_data)

        real_wav = mel_decoder(vocoder, real_data, mean, std)
        fake_wav = mel_decoder(vocoder, fake_data, mean, std)
        cycle_wav = mel_decoder(vocoder, cycle_data, mean, std)

        _, emotion_source_real = emotion_discriminator(real_wav)
        _, emotion_source_fake = emotion_discriminator(fake_wav)
        _, emotion_source_cycle = emotion_discriminator(cycle_wav)

        #g loss
        fake_loss = torch.mean(torch.abs(1 - d_fake_source))
        cycle_loss = torch.mean(torch.abs(real_data - cycle_data))

        #emotion g loss
        fake_emotion_loss = torch.mean(torch.abs(emotion_source_fake - (emotion_source_real + train_target_emotion))) * emotion_loss_size
        cycle_emotion_loss = torch.mean(torch.abs(emotion_source_real - emotion_source_cycle)) * emotion_loss_size / 2
        total_emotion_loss = fake_emotion_loss + cycle_emotion_loss

        #g loss
        g_loss = fake_loss + cycle_loss + total_emotion_loss

        #d loss
        d_real_loss = torch.mean(torch.abs(1 - d_real_source))
        d_fake_loss = torch.mean(torch.abs(0 - d_fake_source))
        d_cycle_loss = torch.mean(torch.abs(0 - d_cycle_source))

        d_loss = d_real_loss + d_fake_loss + d_cycle_loss

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        g_loss.backward(retain_graph=True)
        d_loss.backward()

        g_optimizer.step()
        d_optimizer.step()

    writer.add_scalar("cycle_gan_loss", g_loss, global_step=i)
    writer.add_scalar("cycle_gan_emotion_loss", total_emotion_loss, global_step=i)
    writer.add_scalar("discriminator_loss", d_loss, global_step=i)

    writer.add_image("gan_fake_mel", fake_data, global_step=i)
    writer.add_image("gan_cycle_mel", cycle_data, global_step=i)

    writer.add_audio("gan_fake_audio", fake_wav, global_step=i, sample_rate=fs)
    writer.add_audio("gan_cycle_audio", cycle_wav, global_step=i, sample_rate=fs)

    print("step", i)

    if i % save_pre_step == 0:
        saving_names = ["g", "g2", "d", "g_optimizer", "d_optimizer"]

        for variable_name in saving_names:
            torch.save(vars()[variable_name].state_dict(), os.path.join(model_path, "%s-%d.ckpt" % (variable_name, i)))
