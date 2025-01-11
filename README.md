
## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
$ conda create -n SSL_Spoofing python=3.7
$ conda activate SSL_Spoofing
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
(This fairseq folder can also be downloaded from https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)
$ pip install --editable ./
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are trained on 2019 LA training and evaluated on 2015 evaluation,2019 LA evaluation,2021 LA and DF evaluation and in the wild evaluation database.

[//]: # (The ASVspoof 2019 dataset, which can can be downloaded from [here]&#40;https://datashare.is.ed.ac.uk/handle/10283/3336&#41;.)

[//]: # ()
[//]: # (The ASVspoof 2021 database is released on the zenodo site.)

[//]: # ()
[//]: # (LA [here]&#40;https://zenodo.org/record/4837263#.YnDIinYzZhE&#41;)

[//]: # ()
[//]: # (DF [here]&#40;https://zenodo.org/record/4835108#.YnDIb3YzZhE&#41;)

[//]: # ()
[//]: # (For ASVspoof 2021 dataset keys &#40;labels&#41; and metadata are available [here]&#40;https://www.asvspoof.org/index2021.html&#41;)

## Pre-trained wav2vec 2.0 XLSR (300M)
Download the XLSR models from [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

## Execute Demo
```
cd myresult
python test_dual.py
```  
## Statement
The complete code of the subsequent training will be added after the successful publication of the paper. Thank you for your attention.

