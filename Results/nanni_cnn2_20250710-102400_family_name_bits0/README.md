# nanni_cnn2 4 row matrix family level hiddensize=1024 batch=100

architecture: nanni_cnn2

level: family

coding: 4rowmatrix

command used:

```
PYTHONPATH=$(pwd)/src python run_singlerank_experiment.py --config nanni_cnn2_20250710-202230_family_name_bits0.json 2> nanni_cnn2_$(date +"%Y%m%d-%H%M%S")_family_name_bits0.log
```