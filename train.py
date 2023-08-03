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
steps = 1000
save_pre_step = 100

dataset_path = os.path.join(".", "dataset")
preprocessed_path = os.path.join(".", "preprocess")
model_path = os.path.join(".", "train_model")

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


def tensor_emotion_source(value):
    return torch.tensor(
        emotion_discriminator(cpu_real_data.numpy(), fs)['logits'][0][0], device=device, dtype=torch.float
    )


for i in range(steps):
    for real_data in train_dataloader:
        real_data = real_data.to(device, dtype=torch.float)

        fake_data = g(real_data)
        cycle_data = g2(fake_data)

        d_real_source = d(real_data)
        d_fake_source = d(fake_data)
        d_cycle_source = d(cycle_data)

        cpu_real_data = real_data.detach().cpu()
        cpu_fake_data = fake_data.detach().cpu()
        cpu_cycle_data = cycle_data.detach().cpu()

        real_wav = mel_decoder(vocoder, cpu_real_data, mean, std)
        fake_wav = mel_decoder(vocoder, cpu_fake_data, mean, std)
        cycle_wav = mel_decoder(vocoder, cpu_cycle_data, mean, std)

        emotion_source_real = tensor_emotion_source(cpu_real_data)
        emotion_source_fake = tensor_emotion_source(cpu_fake_data)
        emotion_source_cycle = tensor_emotion_source(cpu_cycle_data)

        #g loss
        fake_loss = torch.mean(torch.abs(1 - d_fake_source))
        cycle_loss = torch.mean(torch.abs(real_data - cycle_data))

        #emotion g loss
        fake_emotion_loss = torch.abs(emotion_source_fake - (emotion_source_real + emotion_modify_level)) * 10
        cycle_emotion_loss = torch.abs(emotion_source_real - emotion_source_cycle) * 5
        total_emotion_loss = fake_emotion_loss + cycle_emotion_loss

        g_loss = fake_loss + cycle_loss + total_emotion_loss

        #d loss
        d_real_loss = torch.mean(torch.abs(1 - d_real_source))
        d_fake_loss = torch.mean(torch.abs(0 - d_fake_source))
        d_cycle_loss = torch.mean(torch.abs(1 - d_cycle_source))

        d_loss = d_real_loss + d_fake_loss + d_cycle_loss

        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        g_loss.backward(retain_graph=True)
        d_loss.backward()

        g_optimizer.step()
        d_optimizer.step()

    print("step", i)
    print("emotion_loss", total_emotion_loss.detach().cpu())
    print("g_loss", g_loss.detach().cpu())
    print("d_loss", d_loss.detach().cpu())

    if i % save_pre_step == 0:
        torch.save(g.state_dict(), os.path.join(model_path, "g-" + str(i) + ".ckpt"))
        torch.save(g2.state_dict(), os.path.join(model_path, "g-" + str(i) + ".ckpt"))
        torch.save(d.state_dict(), os.path.join(model_path, "g-" + str(i) + ".ckpt"))
