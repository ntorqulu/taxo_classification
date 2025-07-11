# nanni_cnn2 4 row matrix family level hiddensize=8 batch=30

architecture: nanni_cnn2

level: family

coding: 4rowmatrix

command used:

```
PYTHONPATH=$(pwd)/src python run_singlerank_experiment.py --config snanni_cnn2_20250708-800030_family_name_bits0.json 2> nanni_cnn2_$(date +"%Y%m%d-%H%M%S")_family_name_bits0.log
```