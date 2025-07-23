# [[ICML2025](https://icml.cc/virtual/2025/poster/45357)] Zero-Shot Cyclic Peptide Design with Composable Geometric Conditions

Codebase is complete but readme is under construction. Directly contact us if you are in hurry to use our codebase.

## Setup

### Environment

The conda environment can be constructed with the configuration `env.yaml`:

```bash
conda env create -f env.yaml
```

The codes are tested with cuda version `12.4` and pytorch version `1.13.1`.

Don't forget to activate the environment before running the codes:

```bash
conda activate Composer
```

## Quick Start(Inference Only)
We offer you a convenient and direct way to generate peptide using the ckpt we provided in `./checkpoint`.

First we should set the condition we want to use:
```bash
export condition={CONDITION}
```
Here, `CONDITION` should be a number:
1: Stapled peptide
2: Head-to-tail peptide
3: Disulfide peptide
4: Bicycle peptide

After that, run the following command:

```bash
python generate.py --config ./configs/pepbench/test_prompt_codesign.yaml --ckpt ./ckpts/LDM_codesign/version_325/checkpoint/epoch37_step85234.ckpt --gpu ${GPU_index}$ --save_dir ./results/${FOLDER NAME}
```

You will get the linear peptide meeting with geometric constraints. You can use the `./evaluate_utils/success_utils.ipynb` to filter the peptide that meet the geometric requirements. 

The last step, please find the corresponding file in relaxer folder to transfer the generated linear peptide into the cyclic peptide. You need to specify the acid amoid index to generate the cyclic structure.

```bash
relaxer/
│   ├── base.py
│   ├── bicycle.py
│   ├── cys_to_cys.py
│   ├── k_to_de.py
│   ├── head_tail.py 
```

## Training from scratch

### Datasets

These datasets are only used for benchmarking models. If you just want to use the trained weights for inferencing on your cases, there is no need to download these datasets.

#### PepBench

1. Download

The datasets, which are originally introduced in this paper, are uploaded to Zenodo at [this url](https://zenodo.org/records/13373108). You can download them as follows:

```bash
mkdir datasets  # all datasets will be put into this directory
wget https://zenodo.org/records/13373108/files/train_valid.tar.gz?download=1 -O ./datasets/train_valid.tar.gz   # training/validation
wget https://zenodo.org/records/13373108/files/LNR.tar.gz?download=1 -O ./datasets/LNR.tar.gz   # test set
wget https://zenodo.org/records/13373108/files/ProtFrag.tar.gz?download=1 -O ./datasets/ProtFrag.tar.gz     # augmentation dataset
```

2. Decompresss

```bash
tar zxvf ./datasets/train_valid.tar.gz -C ./datasets
tar zxvf ./datasets/LNR.tar.gz -C ./datasets
tar zxvf ./datasets/ProtFrag.tar.gz -C ./datasets
```

3. Process

```bash
python -m scripts.data_process.process --index ./datasets/train_valid/all.txt  --out_dir ./datasets/train_valid/processed  # train/validation set
python -m scripts.data_process.process --index ./datasets/LNR/test.txt  --out_dir ./datasets/LNR/processed  # test set
python -m scripts.data_process.process --index ./datasets/ProtFrag/all.txt --out_dir ./datasets/ProtFrag/processed # augmentation dataset
```

The index of processed data for train/validation splits need to be generated as follows, which will result in `datasets/train_valid/processed/train_index.txt` and `datasets/train_valid/processed/valid_index.txt`:

```bash
python -m scripts.data_process.split --train_index datasets/train_valid/train.txt --valid_index datasets/train_valid/valid.txt --processed_dir datasets/train_valid/processed/
```

<!-- #### PepBDB

1. Download

```bash
wget http://huanglab.phys.hust.edu.cn/pepbdb/db/download/pepbdb-20200318.tgz -O ./datasets/pepbdb.tgz
```

2. Decompress

```bash
tar zxvf ./datasets/pepbdb.tgz -C ./datasets/pepbdb
```


3. Process

```bash
python -m scripts.data_process.pepbdb --index ./datasets/pepbdb/peptidelist.txt --out_dir ./datasets/pepbdb/processed
python -m scripts.data_process.split --train_index ./datasets/pepbdb/train.txt --valid_index ./datasets/pepbdb/valid.txt --test_index ./datasets/pepbdb/test.txt --processed_dir datasets/pepbdb/processed/
mv ./datasets/pepbdb/processed/pdbs ./dataset/pepbdb  # re-locate
``` -->

## Model Training

Each task requires the following steps:

1. Train autoencoder
2. Train latent diffusion model
3. Calculate distribution of latent distances between consecutive residues
4. Generation & Evaluation

### Load the pre-trained autoencoder
The pre-trained autoencoder is located at the `./checkpoints/autoencoder.pth`. The weights are directly adopted from paper `Full-atom peptide design with geometric latent diffusion`. 

### Train latent diffusion model
`GPU="0" bash scripts/train.sh ./configs/pepbench/prompt_finetune/train_codesign.yaml`

### Calculate distribution of latent distances between consecutive residues
```bash
python setup_latent_guidance_type.py \
--config ./configs/pepbench/prompt_finetune/setup_prompt_latent_guidance.yaml \
--ckpt ${CKPT_PATH} 
```

## Contact
Thank you for your interest in our work!

Please let us know if you have any questions:
* [jdp22@mails.tsinghua.edu.cn](mailto:jdp22@mails.tsinghua.edu.cn)
* [jdpaerospace2003@gmail.com](mailto:jdpaerospace2003@gmail.com)

## Citations
```bibtex
@misc{jiang2025zeroshot,
    title={Zero-Shot Cyclic Peptide Design with Composable Geometric Conditions},
    author={Dapeng Jiang and Xiangzhe Kong and Jiaqi Han and Mingyu Li and Rui Jiao and Wenbing Huang and Stefano Ermon and Jianzhu Ma and Yang Liu},
    year={2025},
    eprint={2507.04225},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
<!-- ## Quick Links

- [Setup](#setup)
    - [Environment](#environment)
    - [Datasets](#datasets)
    - [Trained Weights](#trained-weights)
- [Usage](#usage)
    - [Peptide Sequence-Structure Co-Design](#peptide-sequence-structure-co-design)
    - [Peptide Binding Structure Prediction](#peptide-binding-structure-prediction)
- [Reproduction of Paper Experiments](#reproduction-of-paper-experiments)
    - [Codesign](#codesign)
    - [Binding Conformation Generation](#binding-conformation-generation)
- [Contact](#contact)
- [Reference](#reference) -->