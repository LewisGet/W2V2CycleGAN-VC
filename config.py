import torch
import os

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
loading_model = 0
emotion_loss_size = 1000

dataset_path = os.path.join(".", "dataset")
preprocessed_path = os.path.join(".", "preprocess")
model_path = os.path.join(".", "train_model")
result_path = os.path.join(".", "train_result")

# Arousal, dominance, valence
train_target_emotion = torch.tensor([0.1, 0.1, 0.1])
