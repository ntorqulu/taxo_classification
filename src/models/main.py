import time
import torch
import torch.nn as nn
import numpy as np
import json
from results import Results, plot_results
from dataset.taxo_dataloaders import TaxoDataLoaders
from model import BasicTaxoModel
from dataset.utils import info, get_default_dataset_path


def train_epoch(results: Results,
                network: nn.Module,
                data_loader: torch.utils.data.DataLoader,
                criterion: torch.nn.functional,
                optimizer: torch.optim.Optimizer,
                device: torch.device
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
               criterion: torch.nn.functional,
               device: torch.device):
    network.eval()
    results.new_epoch()
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target.view(-1))
        results.append(loss.item(), output, target)


def train(hparams: dict, device: torch.device) -> (Results, Results, Results):
    taxo_path = hparams['taxo_path'] if hparams['taxo_path'] else get_default_dataset_path()
    taxo_data_loaders = TaxoDataLoaders(taxo_path=taxo_path,
                                        label_column_name=hparams["label_column_name"],
                                        k=hparams["k"],
                                        bits=hparams["bits"],
                                        batch_size=hparams["batch_size"],
                                        max_rows=hparams["max_rows"])

    network = BasicTaxoModel(
        input_size=taxo_data_loaders.data_length, # 4**k
        hidden_size=hparams['hidden_size'] if hparams['hidden_size'] else taxo_data_loaders.data_length // 2, # TODO:
        output_size=taxo_data_loaders.num_labels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=hparams['learning_rate']) #, weight_decay=hparams.weight_decay)
    #optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    train_results = Results(epochs=hparams['epochs'], name="Train")
    eval_results = Results(epochs=hparams['epochs'], name="Eval")
    test_results = Results(epochs=hparams['epochs'], name="Test")
    for epoch in range(hparams['epochs']):
        info(f"--- Epoch {epoch} ---")
        train_epoch(results=train_results, network=network, data_loader=taxo_data_loaders.train_loader,
                    criterion=criterion, optimizer=optimizer, device=device)
        eval_epoch(results=eval_results, network=network, data_loader=taxo_data_loaders.eval_loader,
                   criterion=criterion, device=device)
        eval_epoch(results=test_results, network=network, data_loader=taxo_data_loaders.train_loader,
                   criterion=criterion, device=device)
        # scheduler.step(eval_results.losses[-1])
    train_results.finish()
    eval_results.finish()
    # eval_epoch(results=test_results, network=network, data_loader=test_loader, criterion=criterion)
    test_results.finish()

    return train_results, eval_results, test_results

def init_device(hparams: dict) -> torch.device:
    np.random.seed(hparams['seed'])
    torch.manual_seed(hparams['seed'])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(hparams['seed'])
        d = torch.device("cuda")
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.manual_seed(hparams['seed'])
        d = torch.device("xpu")
    else:
        d = torch.device("cpu")
    info(f"Device: {d}")
    return d

def run_model(hparams: dict):
    device = init_device(hparams)
    all_results = train(hparams, device)
    for r in all_results:
        info(str(r))
    plot_results(all_results)


def main():
    with open("hparams.json", "r") as f:
        hparams = json.load(f)
    if hparams['taxo_path'] == "":
        hparams['taxo_path'] = get_default_dataset_path()
    t0 = time.time()
    info("Starting")
    run_model(hparams)
    info("Done")
    seconds = time.time() - t0
    minutes = int(seconds / 60)
    seconds = int(seconds - minutes * 60)
    info(f"Elapsed time: {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
