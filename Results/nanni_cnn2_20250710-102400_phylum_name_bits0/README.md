# nanni_cnn2 4 row matrix phylum level hiddensize=1024 batch=100

architecture: nanni_cnn2

level: phylum

coding: 4rowmatrix

command used:

```
PYTHONPATH=$(pwd)/src python run_singlerank_experiment.py --config nanni_cnn2_20250710-230948_phylum_name_bits0.json 2>nanni_cnn2$(date +"%Y%m%d-%H%M%S")_phylum_name_bits0.log 
```