# nanni_cnn1 4 row matrix phylum level

architecture: nanni_cnn1

level: phylum

coding: 4rowmatrix

command used:

```
PYTHONPATH=$(pwd)/src python run_singlerank_experiment.py --config src/models/hyperparams/singlerank/cnn1/nanni_cnn1_4rm_phylum.json 2>nanni_cnn1_4rm_phylum.txt 
```