## Taxonomy DataSet

This module contains two main components:

- `TaxoDataSet`: reads the taxonomy CSV file from disk and implements a cached 
PyTorch [Dataset](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).
- `taxo_data_loaders`: loads the `TaxoDataSet` and returns three 
[DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) instances 
for training, evaluation, and testing.

# TaxoDataSet class

The class implements a cached version to increase performance, using both in-memory and on-disk caches.

When the class is instantiated with a CSV file, it first checks for a memory cache.  If not found, it attempts to 
load a disk cache from a [pickle file](https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html). If that also fails, it loads the data from the CSV file.

The first time the CSV file is loaded (with default parameters), a pickle file is created for future use.  

**Important:** this pickle file is intended for **local use only**. Avoid moving it between systems or Python versions.

The memory cache is at the class level. This means that if you create multiple instances of `TaxoDataSet`,  
they will share the same underlying data. However, each instance will return a different subset depending on the 
initialization parameters.  

# taxo_data_loaders function

This is the main entry point for loading data during modeling.

A tipicall usage will be:

```python
    from dataset import taxo_dataset
    from torch.utils.data import DataLoader

    train: DataLoader
    eval: DataLoader
    test: DataLoader
    train, eval, test = taxo_data_loaders('/tmp/final_taxonomy.csv')
```

