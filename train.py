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


vocoder = torch.hub.load('LewisGet/melgan-neurips', 'load_melgan')

emotion_model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(emotion_model_name)
emotion_discriminator = EmotionModel.from_pretrained(emotion_model_name).to(device=device)


def mel_decoder(vocoder, mel, mel_mean, mel_std):
    denorm_converted = mel * mel_std + mel_mean
    rev = vocoder.inverse(denorm_converted)
    return rev


dataset = VCDataset(load_pickle_file(os.path.join(preprocessed_path, "normalized.pickle")))
dataset_norm_status = np.load(os.path.join(preprocessed_path, "norm_status.npz"))
mean = torch.tensor(dataset_norm_status['mean'].tolist()).to(device=device)
std = torch.tensor(dataset_norm_status['std'].tolist()).to(device=device)
train_target_emotion = train_target_emotion.to(device=device)

train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False)

g = Generator().to(device=device)
g2 = Generator().to(device=device)
d = Discriminator().to(device=device)

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

        fake_data2 = g2(real_data)
        cycle_data2 = g(fake_data2)

        d_real_source = d(real_data)
        d_fake_source = d(fake_data)
        d_cycle_source = d(cycle_data)
        d_fake2_source = d(fake_data2)
        d_cycle2_source = d(cycle_data2)

        real_wav = mel_decoder(vocoder, real_data, mean, std)
        fake_wav = mel_decoder(vocoder, fake_data, mean, std)
        cycle_wav = mel_decoder(vocoder, cycle_data, mean, std)
        fake2_wav = mel_decoder(vocoder, fake_data2, mean, std)
        cycle2_wav = mel_decoder(vocoder, cycle_data2, mean, std)

        _, emotion_source_real = emotion_discriminator(real_wav)
        _, emotion_source_fake = emotion_discriminator(fake_wav)
        _, emotion_source_cycle = emotion_discriminator(cycle_wav)
        _, emotion_source_fake2 = emotion_discriminator(fake2_wav)
        _, emotion_source_cycle2 = emotion_discriminator(cycle2_wav)

        #g loss
        fake_loss = torch.mean(torch.abs(1 - d_fake_source))
        cycle_loss = torch.mean(torch.abs(real_data - cycle_data))

        #g2 loss
        fake2_loss = torch.mean(torch.abs(1 - d_fake2_source))
        cycle2_loss = torch.mean(torch.abs(real_data - cycle_data2))

        #emotion g loss
        fake_emotion_loss = torch.mean(torch.abs(emotion_source_fake - (emotion_source_real + train_target_emotion))) * emotion_loss_size
        cycle_emotion_loss = torch.mean(torch.abs(emotion_source_real - emotion_source_cycle)) * emotion_loss_size / 2

        #emotion g2 loss
        fake2_emotion_loss = torch.mean(torch.abs(emotion_source_fake2 - (emotion_source_real - train_target_emotion))) * emotion_loss_size
        cycle2_emotion_loss = torch.mean(torch.abs(emotion_source_real - emotion_source_cycle2)) * emotion_loss_size / 2

        #total emotion loss
        total_emotion_loss = fake_emotion_loss + cycle_emotion_loss + fake2_emotion_loss + cycle2_emotion_loss

        #g loss
        g_loss = fake_loss + cycle_loss + fake2_loss + cycle2_loss + total_emotion_loss

        #d loss
        d_real_loss = torch.mean(torch.abs(1 - d_real_source))
        d_fake_loss = torch.mean(torch.abs(0 - d_fake_source))
        d_cycle_loss = torch.mean(torch.abs(0 - d_cycle_source))
        d_fake2_loss = torch.mean(torch.abs(0 - d_fake2_source))
        d_cycle2_loss = torch.mean(torch.abs(0 - d_cycle2_source))

        d_loss = d_real_loss * 5.0 + d_fake_loss + d_cycle_loss + d_fake2_loss + d_cycle2_loss

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        g_loss.backward(retain_graph=True)
        d_loss.backward()

        g_optimizer.step()
        d_optimizer.step()

    writer.add_scalar("loss/g/total", g_loss, global_step=i)
    writer.add_scalar("loss/g/g1/emotion", fake_emotion_loss, global_step=i)
    writer.add_scalar("loss/g/g2/emotion", fake2_emotion_loss, global_step=i)
    writer.add_scalar("loss/g/total/emotion", total_emotion_loss, global_step=i)
    writer.add_scalar("loss/discriminator", d_loss, global_step=i)

    writer.add_image("mel/fake/g1", fake_data, global_step=i)
    writer.add_image("mel/cycle/g1", cycle_data, global_step=i)

    writer.add_image("mel/fake/g2", fake_data2, global_step=i)
    writer.add_image("mel/cycle/g2", cycle_data2, global_step=i)

    writer.add_audio("audio/fake/g1", fake_wav, global_step=i, sample_rate=fs)
    writer.add_audio("audio/cycle/g1", cycle_wav, global_step=i, sample_rate=fs)

    writer.add_audio("audio/fake/g2", fake2_wav, global_step=i, sample_rate=fs)
    writer.add_audio("audio/cycle/g2", cycle2_wav, global_step=i, sample_rate=fs)

    print("step", i)

    if i % save_pre_step == 0:
        saving_names = ["g", "g2", "d", "g_optimizer", "d_optimizer"]

        for variable_name in saving_names:
            torch.save(vars()[variable_name].state_dict(), os.path.join(model_path, "%s-%d.ckpt" % (variable_name, i)))
