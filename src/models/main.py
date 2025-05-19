import time
import torch
import torch.nn as nn
import numpy as np
from results import Results, plot_results
from dataset.taxo_dataloader import TaxoDataLoaders
from model import BasicTaxoModel
from dataset.utils import info
from feature_extraction.main import kmer_encoder

HPARAMS = dict(
    taxo_path='/tmp/database.csv',
    max_rows=1000,
    seed=123,
    kmer_k=2,
    epochs=2,
    batch_size=32,
    learning_rate=0.01,
    sequence_encoder=kmer_encoder,
    # input_size= Assigned depending on the dataset,
    hidden_size=None,
    # output_size= Assigned depending on the dataset
)


def init_device() -> torch.device:
    np.random.seed(HPARAMS['seed'])
    torch.manual_seed(HPARAMS['seed'])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(HPARAMS['seed'])
        return torch.device("cuda")
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.manual_seed(HPARAMS['seed'])
        return  torch.device("xpu")
    else:
        return  torch.device("cpu")

device: torch.device = init_device()

def train_epoch(results: Results,
                network: nn.Module,
                data_loader: torch.utils.data.DataLoader,
                criterion: torch.nn.functional,
                optimizer: torch.optim.Optimizer
                ):
    network.train()
    results.new_epoch()
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target.view(-1))
        loss.backward()
        optimizer.step()
        results.append(loss.item(), output, target)


@torch.no_grad()
def eval_epoch(results: Results,
               network: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.functional):
    network.eval()
    results.new_epoch()
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target.view(-1))
        results.append(loss.item(), output, target)


def train(hparams: dict) -> (Results, Results, Results):
    taxo_data_loaders = TaxoDataLoaders(taxo_path=hparams["taxo_path"],
                                        sequence_encoder=hparams["sequence_encoder"],
                                        batch_size=hparams["batch_size"],
                                        max_rows=hparams["max_rows"])

    network = BasicTaxoModel(
        input_size=taxo_data_loaders.data_length,
        hidden_size=hparams['hidden_size'] if hparams['hidden_size'] else taxo_data_loaders.data_length // 2, # TODO:
        output_size=taxo_data_loaders.num_labels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=hparams['learning_rate']) #, weight_decay=hparams.weight_decay)
    #optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # TODO: Scheduller

    train_results = Results(epochs=hparams['epochs'], name="Train")
    eval_results = Results(epochs=hparams['epochs'], name="Eval")
    test_results = Results(epochs=hparams['epochs'], name="Test")
    for epoch in range(hparams['epochs']):
        train_epoch(results=train_results, network=network, data_loader=taxo_data_loaders.train_loader,
                    criterion=criterion, optimizer=optimizer)
        eval_epoch(results=eval_results, network=network, data_loader=taxo_data_loaders.eval_loader,
                   criterion=criterion)
        eval_epoch(results=test_results, network=network, data_loader=taxo_data_loaders.train_loader,
                   criterion=criterion)
    train_results.finish()
    eval_results.finish()
    # eval_epoch(results=test_results, network=network, data_loader=test_loader, criterion=criterion)
    test_results.finish()

    return train_results, eval_results, test_results


def run_model(hparams: dict):
    all_results = train(hparams)
    for r in all_results:
        info(str(r))
    plot_results(all_results)


def main():
    t0 = time.time()
    info("Starting")
    run_model(HPARAMS)
    info("Done")
    seconds = time.time() - t0
    minutes = int(seconds / 60)
    seconds = int(seconds - minutes * 60)
    info(f"Elapsed time: {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
