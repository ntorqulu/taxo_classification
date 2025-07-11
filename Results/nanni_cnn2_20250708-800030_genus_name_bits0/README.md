# nanni_cnn1 4 row matrix genus level hiddensize=8 batch=30

architecture: nanni_cnn2

level: genus

coding: 4rowmatrix

command used:

```
PYTHONPATH=$(pwd)/src python run_singlerank_experiment.py --config nanni_cnn2_20250708-800030_genus_name_bits0.json 2>nnanni_cnn2_$(date +"%Y%m%d-%H%M%S")_genus_name_bits0.log
```