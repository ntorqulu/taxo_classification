# nanni_cnn2 4 row matrix order level hiddensize=8 batch=30

architecture: nanni_cnn2

level: order

coding: 4rowmatrix

command used:

```
PYTHONPATH=$(pwd)/src python run_singlerank_experiment.py --config nanni_cnn2_20250708-800030_order_name_bits0.json 2> nanni_cnn2_$(date +"%Y%m%d-%H%M%S")_order_name_bits0.log
```