# nanni_cnn1 4 row matrix genus level with hidden size of 100

architecture: nanni_cnn1

level: genus

coding: 4rowmatrix

command used:

```
PYTHONPATH=$(pwd)/src python run_singlerank_experiment.py --config src/models/hyperparams/singlerank/cnn1/nanni_cnn1_4rm_genus_hz100.json 2>nanni_cnn1_4rm_genus_hz100.txt 
```