# What this?

this using wav2vec2 to optimize gan.

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

all code about `wav2vec2` copy from [more detail](https://github.com/audeering/w2v2-how-to)
