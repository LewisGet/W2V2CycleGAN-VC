# What this?

this project using `wav2vec2` to supervise generate audio with target emotion vectors.

# Getting start

1. put some train files in `./train_dataset/*.wav`
2. preprocess wav files to mel-spectrogram by runing `preprocess.py`
3. download `wav2vec2` model
4. modifies training option from `train.py`
5. run `train.py`

## About wav2vec2

wav2vec2 pretrain model [link](https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip)

download model and extract demo code.

```python
import os
import audeer

url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
model_path = './model.zip'

audeer.download_url(
    url, 
    model_path, 
    verbose=True,
)

audeer.extract_archive(
    model_path, 
    ".", 
    verbose=True,
)
```

all code and pretrain emotion `wav2vec2` model from:

1. [audeering/w2v2-how-to](https://github.com/audeering/w2v2-how-to)
2. [audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)


## About emotion vectors

1. valence (the pleasantness of a stimulus)
2. arousal (the intensity of emotion provoked by a stimulus)
3. dominance (the degree of control exerted by a stimulus)

[Norms of English lemmas](https://pubmed.ncbi.nlm.nih.gov/23404613/)
[PAD_emotional_state_model](https://en.wikipedia.org/wiki/PAD_emotional_state_model)
