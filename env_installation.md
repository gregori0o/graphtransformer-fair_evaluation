## Setup CONDA

```bash
curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
cd ~
chmod +x miniconda.sh   
./miniconda.sh  

source ~/.bashrc
```

## Setup Python env

```bash
conda install git
git clone https://github.com/gregori0o/graphtransformer-fair_evaluation.git
cd graphtransformer-fair_evaluation

conda create --prefix ./.env python=3.7.4
conda activate ./.env

# for CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch 

# for GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

conda config --append channels conda-forge dglteam
conda install --file  requirements.txt
```

## Download Moleculas dataset

```bash
cd data/
bash script_download_molecules.sh
cd ..
```

## Run experiment

```bash
python perform_experiment.py
```
