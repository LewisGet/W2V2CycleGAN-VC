import torch
import numpy as np
import os

import audonnx

from preprocess import load_pickle_file
from dataset import VCDataset
from model import Discriminator


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

de = Discriminator().to(device=device)

emotion_discriminator = audonnx.load(".")

d_emotion_params = list(de.parameters())
d_emotion_optimizer = torch.optim.Adam(d_emotion_params, lr=d_lr, betas=(0.5, 0.999))


def tensor_emotion_source(value):
    return torch.tensor(
        emotion_discriminator(cpu_real_data.numpy(), fs)['logits'][0][0], device=device, dtype=torch.float
    )


for i in range(steps):
    for real_data in train_dataloader:
        real_data = real_data.to(device, dtype=torch.float)
        fake_data = torch.randn_like(real_data, dtype=torch.float).to(device)

        d_emotion_source_real = de(real_data)
        d_emotion_source_fake = de(fake_data)

        cpu_real_data = real_data.detach().cpu()
        cpu_fake_data = fake_data.detach().cpu()

        real_wav = mel_decoder(vocoder, cpu_real_data, mean, std)
        real_wav = mel_decoder(vocoder, cpu_fake_data, mean, std)

        emotion_source_real = tensor_emotion_source(cpu_real_data)
        emotion_source_fake = tensor_emotion_source(cpu_fake_data)

        de_real_loss = torch.mean(torch.abs(emotion_source_real - d_emotion_source_real))
        de_fake_loss = torch.mean(torch.abs(emotion_source_fake - d_emotion_source_fake))

        de_loss = de_real_loss + de_fake_loss

        d_emotion_optimizer.zero_grad()
        de_loss.backward()
        d_emotion_optimizer.step()

    print("step", i)
    print("d_loss", de_loss.detach().cpu())

    if i % save_pre_step == 0:
        torch.save(de.state_dict(), os.path.join(model_path, "d-" + str(i) + ".ckpt"))
        torch.save(d_emotion_optimizer.state_dict(), os.path.join(model_path, "d-emotion-optimizer-" + str(i) + ".ckpt"))
