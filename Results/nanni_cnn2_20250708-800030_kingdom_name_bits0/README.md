# nanni_cnn2 4 row matrix kingdom level hiddensize=8 batch=30

architecture: nanni_cnn2

level: kingdom

coding: 4rowmatrix

command used:

```
PYTHONPATH=$(pwd)/src python run_singlerank_experiment.py --config nanni_cnn2_20250708-800030_kingdom_name_bits0.json 2> nanni_cnn2_$(date +"%Y%m%d-%H%M%S")_kingdom_name_bits0.log
```