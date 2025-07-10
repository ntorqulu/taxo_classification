# nanni_cnn2 4 row matrix kingdom level hiddensize=8 batch=30

architecture: nanni_cnn2

level: kingdom

coding: 4rowmatrix

command used:

```
PYTHONPATH=$(pwd)/src python run_singlerank_experiment.py --config src/models/hyperparams/singlerank/cnn1/nanni_cnn1_4rm_phylum.json 2>nanni_cnn1_4rm_phylum.txt 
```