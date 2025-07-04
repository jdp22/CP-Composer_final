# [[ICML2025](https://icml.cc/virtual/2025/poster/45357)] Zero-Shot Cyclic Peptide Design with Composable Geometric Conditions

Codebase is complete but readme is under construction. Directly contact us if you are in hurry to use our codebase.

### (Optional) Datasets

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

#### PepBDB

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
```

## Contact
Thank you for your interest in our work!

Please let us know if you have any questions:
* [jdp22@mails.tsinghua.edu.cn](mailto:jdp22@mails.tsinghua.edu.cn)
* [jdpaerospace2003@gmail.com](mailto:jdpaerospace2003@gmail.com)
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