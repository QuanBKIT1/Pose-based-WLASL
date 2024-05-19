# Pose-based method in WLASL2000 dataset

## Dataset Preparation

WLASL Dataset HomePage: https://dxli94.github.io/WLASL/

WLASL is the largest video dataset for Word-Level American Sign Language (ASL) recognition, which features 2,000 common different words in ASL

Split 4 subsets $K=100,300,1000,2000$ . Statistics on training, validation and testing on each subset of the WLASL dataset:

| Datasets  | Gloss | Videos | Mean  | Train | Val  | Test |
| --------- | ----- | ------ | ----- | ----- | ---- | ---- |
| WLASL100  | 100   | 2038   | 20.38 | 1442  | 338  | 258  |
| WLASL300  | 300   | 5117   | 17.06 | 3549  | 900  | 668  |
| WLASL1000 | 1000  | 13168  | 13.17 | 8974  | 2318 | 1876 |
| WLASL2000 | 2000  | 21083  | 10.53 | 14289 | 3916 | 2878 |

## Data preprocessing

## Model

### 1. ST-GCN

Link paper: https://arxiv.org/abs/1709.04875

### 2. ST-GCN++

Link paper: https://arxiv.org/abs/2205.09443

### 3. CTR-GCN

Link paper: https://arxiv.org/abs/2107.12213
