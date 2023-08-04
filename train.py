import torch
import torchaudio
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
ga_lr = 3e-4
d_lr = 1e-4
# we can test 0.2 for two steps modify
emotion_modify_level = 0.4
fs = 16000
steps = 1000
save_pre_step = 100

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
ga = Generator().to(device=device)
d = Discriminator().to(device=device)
de = Discriminator().to(device=device)

emotion_discriminator = audonnx.load(".")

g_params = list(g.parameters()) + list(g2.parameters())
ga_params = list(ga.parameters())
d_params = list(d.parameters())
d_emotion_params = list(de.parameters())

g_optimizer = torch.optim.Adam(g_params, lr=g_lr, betas=(0.5, 0.999))
ga_optimizer = torch.optim.Adam(g_params, lr=ga_lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(d_params, lr=d_lr, betas=(0.5, 0.999))
d_emotion_optimizer = torch.optim.Adam(d_emotion_params, lr=d_lr, betas=(0.5, 0.999))


def tensor_emotion_source(value):
    return torch.tensor(
        emotion_discriminator(cpu_real_data.numpy(), fs)['logits'][0][0], device=device, dtype=torch.float
    )


for i in range(steps):
    for real_data in train_dataloader:
        real_data = real_data.to(device, dtype=torch.float)

        fake_data = g(real_data)
        cycle_data = g2(fake_data)
        fake_data_a = ga(real_data)

        d_real_source = d(real_data)
        d_fake_source = d(fake_data)
        d_cycle_source = d(cycle_data)
        d_fake_source_a = d(fake_data_a)

        d_emotion_source_real = de(real_data)
        d_emotion_source_fake = de(fake_data)
        d_emotion_source_cycle = de(cycle_data)
        d_emotion_source_fake_a = de(fake_data_a)

        cpu_real_data = real_data.detach().cpu()
        cpu_fake_data = fake_data.detach().cpu()
        cpu_cycle_data = cycle_data.detach().cpu()
        cpu_fake_data_a = fake_data_a.detach().cpu()

        real_wav = mel_decoder(vocoder, cpu_real_data, mean, std)
        fake_wav = mel_decoder(vocoder, cpu_fake_data, mean, std)
        cycle_wav = mel_decoder(vocoder, cpu_cycle_data, mean, std)
        fake_wav_a = mel_decoder(vocoder, cpu_fake_data_a, mean, std)

        emotion_source_real = tensor_emotion_source(cpu_real_data)
        emotion_source_fake = tensor_emotion_source(cpu_fake_data)
        emotion_source_cycle = tensor_emotion_source(cpu_cycle_data)
        emotion_source_fake_a = tensor_emotion_source(cpu_fake_data_a)

        #g loss
        fake_loss = torch.mean(torch.abs(1 - d_fake_source))
        cycle_loss = torch.mean(torch.abs(real_data - cycle_data))

        #ga loss
        fake_loss_a = torch.mean(torch.abs(1 - d_fake_source_a))

        #emotion g loss
        fake_emotion_loss = torch.mean(torch.abs(d_emotion_source_fake - (d_emotion_source_real + emotion_modify_level))) * 10
        cycle_emotion_loss = torch.mean(torch.abs(d_emotion_source_real - d_emotion_source_cycle)) * 5
        total_emotion_loss = fake_emotion_loss + cycle_emotion_loss

        #emotion ga loss
        fake_emotion_loss_a = torch.mean(torch.abs(d_emotion_source_fake_a - (d_emotion_source_real + emotion_modify_level))) * 10

        #cycle gan compare with gan source
        g_vs_ga_loss = torch.mean(fake_loss - fake_loss_a) + torch.mean(fake_emotion_loss - fake_emotion_loss_a)
        ga_vs_g_loss = torch.mean(fake_loss_a - fake_loss) + torch.mean(fake_emotion_loss_a - fake_emotion_loss)

        #g loss
        g_loss = fake_loss + cycle_loss + total_emotion_loss + g_vs_ga_loss

        #ga loss
        ga_loss = fake_loss_a + fake_emotion_loss_a + ga_vs_g_loss

        #d loss
        d_real_loss = torch.mean(torch.abs(1 - d_real_source))
        d_fake_loss = torch.mean(torch.abs(0 - d_fake_source))
        d_cycle_loss = torch.mean(torch.abs(0 - d_cycle_source))
        d_fake_loss_a = torch.mean(torch.abs(0 - d_fake_source_a))

        d_loss = d_real_loss + d_fake_loss + d_cycle_loss + d_fake_loss_a

        #d emotion loss
        de_real_loss = torch.mean(torch.abs(emotion_source_real - d_emotion_source_real))
        de_fake_loss = torch.mean(torch.abs(emotion_source_fake - d_emotion_source_fake))
        de_cycle_loss = torch.mean(torch.abs(emotion_source_cycle - d_emotion_source_cycle))
        de_fake_loss_a = torch.mean(torch.abs(emotion_source_fake_a - d_emotion_source_fake_a))

        de_loss = de_real_loss + de_fake_loss + de_cycle_loss + de_fake_loss_a

        g_optimizer.zero_grad()
        ga_optimizer.zero_grad()
        d_optimizer.zero_grad()
        d_emotion_optimizer.zero_grad()

        g_loss.backward(retain_graph=True)
        ga_loss.backward(retain_graph=True)
        de_loss.backward(retain_graph=True)
        d_loss.backward()

        g_optimizer.step()
        ga_optimizer.step()
        d_emotion_optimizer.step()
        d_optimizer.step()

    print("step", i)
    print("emotion_loss", total_emotion_loss.detach().cpu())
    print("g_loss", g_loss.detach().cpu())
    print("ga_loss", ga_loss.detach().cpu())
    print("d_loss", d_loss.detach().cpu())
    print("de_loss", de_loss.detach().cpu())

    if i % save_pre_step == 0:
        torch.save(g.state_dict(), os.path.join(model_path, "g-" + str(i) + ".ckpt"))
        torch.save(g2.state_dict(), os.path.join(model_path, "g2-" + str(i) + ".ckpt"))
        torch.save(ga.state_dict(), os.path.join(model_path, "ga-" + str(i) + ".ckpt"))
        torch.save(d.state_dict(), os.path.join(model_path, "d-" + str(i) + ".ckpt"))
        torch.save(de.state_dict(), os.path.join(model_path, "de-" + str(i) + ".ckpt"))
        torch.save(g_optimizer.state_dict(), os.path.join(model_path, "g-optimizer-" + str(i) + ".ckpt"))
        torch.save(ga_optimizer.state_dict(), os.path.join(model_path, "ga-optimizer-" + str(i) + ".ckpt"))
        torch.save(d_optimizer.state_dict(), os.path.join(model_path, "d-optimizer-" + str(i) + ".ckpt"))
        torch.save(d_emotion_optimizer.state_dict(), os.path.join(model_path, "d-emotion-optimizer-" + str(i) + ".ckpt"))

        for wav_name in ["fake_wav", "cycle_wav", "fake_wav_a"]:
            torchaudio.save(os.path.join(result_path, "%s_%d.wav" % (wav_name, i)), vars()[wav_name].detach().cpu(), sample_rate=fs)
