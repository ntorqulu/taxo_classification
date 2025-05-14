import torch
from torch.utils.data import DataLoader
from taxo_dataset import TaxoDataset

def taxo_data_loaders(taxo_path: str,
                      batch_size: int,
                      max_rows: int | float = 1.) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get DataLoaders for training, evaluation, and testing datasets.

    This function creates and returns three PyTorch DataLoaders corresponding to training,
    evaluation, and testing datasets. Datasets are loaded from a specified taxonomy file
    using the TaxoDataset class. DataLoaders are configured with the given batch size and
    no shuffling to optimize performance. The returned DataLoaders enable efficient
    mini-batch data handling for model training, validation, and testing.

    Parameters
    ----------
    taxo_path : str
        The file path to the taxonomy dataset, which contains data for all splits
        (train, eval, test).
    batch_size : int
        The number of data samples to be included in each batch.
    max_rows : int | float, optional
        The maximum number of rows to load from the dataset. See TaxoDataSet docstring

    Returns
    -------
    tuple of DataLoader
        A tuple containing three DataLoaders for training, evaluation, and testing datasets.
    """
    train_dataset = TaxoDataset(taxo_path=taxo_path, split='train', max_rows=max_rows)
    eval_dataset = train_dataset.new_split('eval')
    test_dataset = train_dataset.new_split('test')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False, # We do the suffle on the DataLoader to increase performance
    )

    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False, # We do the suffle on the DataLoader to increase performance
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, # We do the suffle on the DataLoader to increase performance
    )

    return train_loader, eval_loader, test_loader
