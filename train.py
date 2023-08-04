import torch
import torchaudio
import pickle
import numpy as np
import os

import audonnx

from preprocess import load_pickle_file
from dataset import VCDataset
from model import Generator, Discriminator

from torch.utils.tensorboard import SummaryWriter


vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')


def mel_decoder(vocoder, mel, mel_mean, mel_std):
    denorm_converted = mel * mel_std + mel_mean
    rev = vocoder.inverse(denorm_converted)
    return rev


torch.cuda.set_device(0)
device = "cuda"
g_lr = 2e-4
ga_lr = 3e-4
d_lr = 1e-4
# we can test 0.2 for two steps modify
emotion_modify_level = 0.4
fs = 16000
steps = 1000
save_pre_step = 100
enable_two_gan_compete = False

dataset_path = os.path.join(".", "dataset")
preprocessed_path = os.path.join(".", "preprocess")
model_path = os.path.join(".", "train_model")
result_path = os.path.join(".", "train_result")

dataset = VCDataset(load_pickle_file(os.path.join(preprocessed_path, "normalized.pickle")))
dataset_norm_status = np.load(os.path.join(preprocessed_path, "norm_status.npz"))
mean = dataset_norm_status['mean']
std = dataset_norm_status['std']

train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False)

g = Generator().to(device=device)
g2 = Generator().to(device=device)
d = Discriminator().to(device=device)
de = Discriminator().to(device=device)

emotion_discriminator = audonnx.load(".")

g_params = list(g.parameters()) + list(g2.parameters())
d_params = list(d.parameters())
d_emotion_params = list(de.parameters())

g_optimizer = torch.optim.Adam(g_params, lr=g_lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(d_params, lr=d_lr, betas=(0.5, 0.999))
d_emotion_optimizer = torch.optim.Adam(d_emotion_params, lr=d_lr, betas=(0.5, 0.999))

if enable_two_gan_compete:
    ga = Generator().to(device=device)
    ga_params = list(ga.parameters())
    ga_optimizer = torch.optim.Adam(ga_params, lr=ga_lr, betas=(0.5, 0.999))

def tensor_emotion_source(value):
    return torch.tensor(
        emotion_discriminator(cpu_real_data.numpy(), fs)['logits'][0][0], device=device, dtype=torch.float
    )


writer = SummaryWriter()

for i in range(steps):
    for real_data in train_dataloader:
        real_data = real_data.to(device, dtype=torch.float)

        fake_data = g(real_data)
        cycle_data = g2(fake_data)

        if enable_two_gan_compete:
            fake_data_a = ga(real_data)

        d_real_source = d(real_data)
        d_fake_source = d(fake_data)
        d_cycle_source = d(cycle_data)

        if enable_two_gan_compete:
            d_fake_source_a = d(fake_data_a)

        d_emotion_source_real = de(real_data)
        d_emotion_source_fake = de(fake_data)
        d_emotion_source_cycle = de(cycle_data)

        if enable_two_gan_compete:
            d_emotion_source_fake_a = de(fake_data_a)

        cpu_real_data = real_data.detach().cpu()
        cpu_fake_data = fake_data.detach().cpu()
        cpu_cycle_data = cycle_data.detach().cpu()

        if enable_two_gan_compete:
            cpu_fake_data_a = fake_data_a.detach().cpu()

        real_wav = mel_decoder(vocoder, cpu_real_data, mean, std)
        fake_wav = mel_decoder(vocoder, cpu_fake_data, mean, std)
        cycle_wav = mel_decoder(vocoder, cpu_cycle_data, mean, std)

        if enable_two_gan_compete:
            fake_wav_a = mel_decoder(vocoder, cpu_fake_data_a, mean, std)

        emotion_source_real = tensor_emotion_source(cpu_real_data)
        emotion_source_fake = tensor_emotion_source(cpu_fake_data)
        emotion_source_cycle = tensor_emotion_source(cpu_cycle_data)

        if enable_two_gan_compete:
            emotion_source_fake_a = tensor_emotion_source(cpu_fake_data_a)

        #g loss
        fake_loss = torch.mean(torch.abs(1 - d_fake_source))
        cycle_loss = torch.mean(torch.abs(real_data - cycle_data))

        #ga loss
        if enable_two_gan_compete:
            fake_loss_a = torch.mean(torch.abs(1 - d_fake_source_a))

        #emotion g loss
        fake_emotion_loss = torch.mean(torch.abs(d_emotion_source_fake - (d_emotion_source_real + emotion_modify_level))) * 10
        cycle_emotion_loss = torch.mean(torch.abs(d_emotion_source_real - d_emotion_source_cycle)) * 5
        total_emotion_loss = fake_emotion_loss + cycle_emotion_loss

        #emotion ga loss
        if enable_two_gan_compete:
            fake_emotion_loss_a = torch.mean(torch.abs(d_emotion_source_fake_a - (d_emotion_source_real + emotion_modify_level))) * 10

        #cycle gan compare with gan source
        if enable_two_gan_compete:
            g_vs_ga_loss = torch.mean(fake_loss - fake_loss_a) + torch.mean(fake_emotion_loss - fake_emotion_loss_a)
            ga_vs_g_loss = torch.mean(fake_loss_a - fake_loss) + torch.mean(fake_emotion_loss_a - fake_emotion_loss)

        #g loss
        g_loss = fake_loss + cycle_loss + total_emotion_loss

        if enable_two_gan_compete:
            g_loss = fake_loss + cycle_loss + total_emotion_loss + g_vs_ga_loss

        #ga loss
        if enable_two_gan_compete:
            ga_loss = fake_loss_a + fake_emotion_loss_a + ga_vs_g_loss

        #d loss
        d_real_loss = torch.mean(torch.abs(1 - d_real_source))
        d_fake_loss = torch.mean(torch.abs(0 - d_fake_source))
        d_cycle_loss = torch.mean(torch.abs(0 - d_cycle_source))

        if enable_two_gan_compete:
            d_fake_loss_a = torch.mean(torch.abs(0 - d_fake_source_a))

        d_loss = d_real_loss + d_fake_loss + d_cycle_loss

        if enable_two_gan_compete:
            d_loss = d_real_loss + d_fake_loss + d_cycle_loss + d_fake_loss_a

        #d emotion loss
        de_real_loss = torch.mean(torch.abs(emotion_source_real - d_emotion_source_real))
        de_fake_loss = torch.mean(torch.abs(emotion_source_fake - d_emotion_source_fake))
        de_cycle_loss = torch.mean(torch.abs(emotion_source_cycle - d_emotion_source_cycle))

        if enable_two_gan_compete:
            de_fake_loss_a = torch.mean(torch.abs(emotion_source_fake_a - d_emotion_source_fake_a))

        de_loss = de_real_loss + de_fake_loss + de_cycle_loss

        if enable_two_gan_compete:
            de_loss = de_real_loss + de_fake_loss + de_cycle_loss + de_fake_loss_a

        g_optimizer.zero_grad()

        if enable_two_gan_compete:
            ga_optimizer.zero_grad()

        d_optimizer.zero_grad()
        d_emotion_optimizer.zero_grad()

        g_loss.backward(retain_graph=True)

        if enable_two_gan_compete:
            ga_loss.backward(retain_graph=True)

        de_loss.backward(retain_graph=True)
        d_loss.backward()

        g_optimizer.step()

        if enable_two_gan_compete:
            ga_optimizer.step()

        d_emotion_optimizer.step()
        d_optimizer.step()

    writer.add_scalar("cycle_gan_loss", g_loss, global_step=i)

    if enable_two_gan_compete:
        writer.add_scalar("gan_loss", ga_loss, global_step=i)

    writer.add_scalar("cycle_gan_emotion_loss", total_emotion_loss, global_step=i)
    writer.add_scalar("discriminator_loss", d_loss, global_step=i)
    writer.add_scalar("emotion_discriminator_loss", de_loss, global_step=i)

    writer.add_image("gan_fake_mel", fake_data, global_step=i)
    writer.add_image("gan_cycle_mel", cycle_data, global_step=i)

    if enable_two_gan_compete:
        writer.add_image("gan_fake_without_cycle_mel", fake_data_a, global_step=i)

    writer.add_audio("gan_fake_audio", fake_wav, global_step=i, sample_rate=fs)
    writer.add_audio("gan_cycle_audio", cycle_wav, global_step=i, sample_rate=fs)

    if enable_two_gan_compete:
        writer.add_audio("gan_fake_without_cycle_audio", fake_wav_a, global_step=i, sample_rate=fs)

    print("step", i)
    print("emotion_loss", total_emotion_loss.detach().cpu())
    print("g_loss", g_loss.detach().cpu())

    if enable_two_gan_compete:
        print("ga_loss", ga_loss.detach().cpu())

    print("d_loss", d_loss.detach().cpu())
    print("de_loss", de_loss.detach().cpu())

    if i % save_pre_step == 0:
        saving_names = ["g", "g2", "d", "de", "g_optimizer", "d_optimizer", "d_emotion_optimizer"]

        if enable_two_gan_compete:
            saving_names = ["g", "g2", "ga", "d", "de", "g_optimizer", "ga_optimizer", "d_optimizer", "d_emotion_optimizer"]

        for variable_name in saving_names:
            torch.save(vars()[variable_name].state_dict(), os.path.join(model_path, "%s-%d.ckpt" % (variable_name, i)))

        saving_wav_names = ["fake_wav", "cycle_wav"]

        if enable_two_gan_compete:
            saving_wav_names = ["fake_wav", "cycle_wav", "fake_wav_a"]

        for wav_name in saving_wav_names:
            torchaudio.save(os.path.join(result_path, "%s_%d.wav" % (wav_name, i)), vars()[wav_name].detach().cpu(), sample_rate=fs)
