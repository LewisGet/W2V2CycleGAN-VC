# What this?

this using wev2vec2 to optimize gan.

## About wev2vec2

wev2vec2 pretrain model [link](https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip)

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

all code about `wev2vec2` copy from [more detail](https://github.com/audeering/w2v2-how-to)
