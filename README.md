# taxo_classification

## Introduction
With the advent of the new DNA sequencing thecnologies, methodologies based on such thecniques have unveiled a bast field of applications to infer the Biodiversity of the ecosystems to monitorize such environments and their habitats. Among these thecniques DNA metabarcoding has emerged as a powerfull tool for biomonitoring. This thechnique relies on sequencing a targuet region of the genome of thousands of individuals at the same time and their latter assignment to the different taxa. However, one of the main constrains of the method is this taxonomic assignment due to the missing information in the databases and the amount of resources required to compute such analysis. While the latter problem has been assessed by using bigger computing services and optimizing the algorithms, to be able to run such models in the field withouth internet connection is still a challange. 

Here we propose Deep Learning models as a potential solution for such problem. While the training process is resource demanding, deppending on the architecture of the module, the predictions can be potentially manageable. In this project we explore the following bottlenecks of the implementation of this thecnology in the field of Biomonitoring:

### DNA codification:
We explored three different approaches:
- kmerisation
- 4xN matrix
- One-hot-encoding

### Model architechture
- Fully connected
- CNN

### Category Balance for training

### Hyperparameter optimisation

## Methodology
To obtained the results and to replicate them here we present the different steps followed for the analysis

### Obtain Database
To obtain the database we used the sequences from [mkCOInr](https://github.com/meglecz/mkCOInr) as described in [NJORDR-MJOLNIR3](https://github.com/adriantich/NJORDR-MJOLNIR3) and can be downloaded from google drive: [NJORDR_sequences](https://drive.google.com/file/d/1YU_jIRIm9rpEC4okD5xh2qr3EnGBg3i8/view?usp=sharing), [names.dmp](https://drive.google.com/file/d/1WrRHX5Mf23ijg03K5YNaAx3dtzgIX5Zn/view?usp=sharing) and [nodes.dmp](https://drive.google.com/file/d/1D4g7PP-mdP9xqsxM9ZC_Bz9wqANkf6UN/view?usp=sharing). These are sequences from the public databases NCBI and BOLD and sequences obtained by scientific groups in University of Barcelona, Center for Advanced Studies of Blanes and Alfred Wagener Institute.

### Preprocessing

To clean and format the database, the following steps were performed:

#### 1. Format Raw Data with `SequenceFormatter` class
- Merge the taxonomic information and cut the region defined by the Leray-XT primers using the `SequenceFormatter` class.
- For each sequence, retain only the information regarding the following ranks: **Superkingdom**, **Kingdom**, **Phylum**, **Order**, **Species**.
- If any sequence has one of these ranks empty, retrieve the information from the rank immediately below (or two levels below) and mark it as *predicted* for further standardization or removal.
- Capitalize all bases in the sequence.

#### 2. Clean Data with `TaxonomyDataCleaner` class
- Remove sequences shorter than **299 bp**.
- Filter out sequences with ambiguous bases (e.g., **N**).
- Filter out sequences with non-standard bases.
- Enforce taxonomy consistency across ranks: if two sequences are identical but their taxonomy assignments differ (excluding blanks), remove them as inconclusive.
- Remove duplicate sequences.
- Ensure complete taxonomic information up to the **species** level.

#### 3. Create Hierarchical Dataset with `TaxonomyDataFilter` class
- Filter sequences longer than **320 bp**.
- Clean approximated or uncertain taxonomic names, removing or standardizing them.
- Create four nested classification levels:

  - **Level 1 (Kingdom)**:  
    `Metazoa`, `Viridiplantae`, `Fungi`, `Other_euk`, `No_euk`

  - **Level 2 (Phylum)**:  
    `Arthropoda`, `Chordata`, `Mollusca`, `Annelida`, `Echinodermata`, `Platyhelminthes`, `Cnidaria`, `Other_metazoa`, `No_metazoa`

  - **Level 3 (Class)**:  
    `Insecta`, `Arachnida`, `Malacostraca`, `Collembola`, `Hexanauplia`, `Thecostraca`, `Branchiopoda`, `Diplopoda`, `Ostracoda`, `Chilopoda`, `Pycnogonida`, `Other_arthropoda`, `No_arthropoda`

  - **Level 4 (Order)**:  
    `Diptera`, `Lepidoptera`, `Hymenoptera`, `Coleoptera`, `Hemiptera`, `Trichoptera`, `Orthoptera`, `Ephemeroptera`, `Odonata`, `Blattodea`, `Thysanoptera`, `Psocoptera`, `Plecoptera`, `Neuroptera`, `Other_insecta`, `No_insecta`

---

To replicate the creation of the database, run the following command from a Unix terminal:

```bash
cd taxo_classification
python -m src.preprocessing.filter
```

### Testing the architectures

The project supports two main types of models:

#### Single-Rank Models
These models predict a single taxonomic level (e.g., only order_name or only phylum_name).

**Available Model Types:**
- `basic`: Simple MLP model
- `enhanced_mlp`: Enhanced MLP with configurable layers and batch normalization
- `cnn`: CNN model for sequence-based classification
- `nanni_cnn1`: CNN model from Nanni et al. 2024 (version 1)
- `nanni_cnn2`: CNN model from Nanni et al. 2024 (version 2)
- `nanni_att`: Attention-based model from Nanni et al. 2024
- `nanni_att_kmer`: Attention-based model for k-mer encoded data
- `bert`: BERT-based model for taxonomy classification

**Running Single-Rank Models:**
```bash
cd taxo_classification
python run_singlerank_experiment.py --config src/models/hyperparams/singlerank/nanni_cnn1_hparams.json
```

#### Multi-Rank Models
These models predict multiple taxonomic levels simultaneously, leveraging hierarchical relationships.

**Available Model Types:**
- `hierarchical`: Hierarchical model with shared feature extractor and multiple output heads
- `cascade_hierarchical`: Cascade model where predictions flow from higher to lower levels
- `gnn_hierarchical`: Graph Neural Network model for hierarchical taxonomy classification

**Running Multi-Rank Models:**
```bash
cd taxo_classification
python run_multirank_experiment.py --config src/models/hyperparams/multirank/hierarchical_genus_hparams.json
```

#### Hyperparameters JSON Configuration

Both single-rank and multi-rank models use JSON configuration files to specify hyperparameters. The configuration files are located in:
- Single-rank: `src/models/hyperparams/singlerank/`
- Multi-rank: `src/models/hyperparams/multirank/`

**Common Hyperparameters (Single-Rank and Multi-Rank):**

- `batch_size`: int, size of the batch for the training process
- `epochs`: int, number of epochs to use in the training phase
- `learning_rate`: float, learning rate used for the optimizer (default: 0.001 for Adam, 0.01 for SGD)
- `seed`: int, seed for reproducibility (default: 42)
- `dataset_name`: str, name of the dataset to use ("filtered_ranks" or "all_ranks")
- `label_column_name`: str, target taxonomic level for single-rank models or highest level for multi-rank models
- `max_rows`: float or int, proportion or total number of sequences to use
- `patience`: int, early stopping patience (epochs with no improvement, default: 5)
- `fast_mode`: bool, use faster evaluation with minimal metrics
- `eval_frequency`: int, how often to run full evaluation (epochs)

**Single-Rank Specific Hyperparameters:**

- `model_type`: str, model architecture type
- `k`: int, size of the window for k-merization (if specified, bits must be null)
- `bits`: int, number of bits for one-hot encoding (if specified, k must be null)
- `hidden_size`: int, hidden layer size for basic and enhanced_mlp models
- `sequence_length`: int, input sequence length for Nanni models (default: 313)
- `dropout`: float, dropout probability
- `use_batch_norm`: bool, whether to use batch normalization
- `optimizer`: str, optimizer type ("adam" or "sgd")
- `use_scheduler`: bool, whether to use learning rate scheduler
- `scheduler`: str, scheduler type ("plateau", "cosine", or "by_step")
- `weight_decay`: float, weight decay for optimizers

**Multi-Rank Specific Hyperparameters:**

- `model_type`: str, multi-rank model architecture type
- `shared_hidden_sizes`: list[int], hidden layer sizes for shared feature extractor
- `level_specific_sizes`: dict, mapping of taxonomic levels to specific hidden layer sizes
- `level_weights`: dict, weights for different taxonomic levels in loss calculation
- `dropout`: float, dropout probability
- `use_confidence_weighting`: bool, whether to use confidence weighting (cascade models)
- `cascade_weight`: float, weight for cascade consistency loss
- `confidence_weight`: float, weight for confidence regularization
- `gnn_layers`: int, number of GNN layers (GNN models)
- `use_attention`: bool, whether to use attention mechanism (GNN models)
- `graph_weight`: float, weight for graph structure regularization (GNN models)
- `consistency_weight`: float, weight for hierarchical consistency (GNN models)

**Example Configuration Files:**

Single-rank example (`nanni_cnn1_hparams.json`):
```json
{
    "parquets_path": "",
    "dataset_name": "filtered_ranks",
    "max_rows": 1.0,
    "seed": 123,
    "epochs": 15,
    "batch_size": 30,
    "learning_rate": 0.001,
    "weight_decay": 0,
    "label_column_name": "order_name",
    "k": null,
    "bits": 0,
    "hidden_size": 8,
    "model_type": "nanni_cnn1",
    "optimizer": "adam",
    "use_scheduler": true,
    "scheduler": "by_step",
    "every_n_epochs": 50,
    "seq_len_filter": 313,
    "dropout": 0.5,
    "use_batch_norm": true,
    "fast_mode": true,
    "eval_frequency": 5
}
```

Multi-rank example (`hierarchical_genus_hparams.json`):
```json
{
    "parquets_path": "",
    "dataset_name": "all_ranks",
    "max_rows": 1.0,
    "seed": 123,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "label_column_name": "genus_name",
    "k": null,
    "bits": 4,
    "model_type": "hierarchical",
    "shared_hidden_sizes": [512, 256],
    "level_specific_sizes": {
        "kingdom_name": [128, 64],
        "phylum_name": [128, 64],
        "class_name": [128, 64],
        "order_name": [128, 64],
        "family_name": [128, 64],
        "genus_name": [128, 64]
    },
    "level_weights": {
        "kingdom_name": 1.0,
        "phylum_name": 1.0,
        "class_name": 1.0,
        "order_name": 1.0,
        "family_name": 1.0,
        "genus_name": 1.0
    },
    "dropout": 0.3,
    "patience": 10,
    "fast_mode": false,
    "eval_frequency": 1
}
```

**Available Configuration Files:**

Single-rank configurations:
- `basic_hparams.json` - Basic MLP model
- `enhanced_mlp_hparams.json` - Enhanced MLP with batch normalization
- `cnn_hparams.json` - CNN model
- `nanni_cnn1_hparams.json` - Nanni CNN1 model
- `nanni_cnn2_hparams.json` - Nanni CNN2 model
- `nanni_att_hparams.json` - Nanni attention model
- `nanni_att_2mer_hparams.json` - Nanni attention model for 2-mer encoding
- `bert_hparams.json` - BERT model
- `kmer_hparams.json` - K-mer encoding configuration
- `one_hot_hparams.json` - One-hot encoding configuration

Multi-rank configurations:
- `hierarchical_hparams.json` - Basic hierarchical model
- `hierarchical_genus_hparams.json` - Hierarchical model up to genus level
- `hierarchical_species_hparams.json` - Hierarchical model up to species level
- `cascade_hparams.json` - Cascade hierarchical model
- `cascade_genus_hparams.json` - Cascade model up to genus level
- `cascade_species_hparams.json` - Cascade model up to species level
- `gnn_hierarchical_hparams.json` - GNN hierarchical model
- `gnn_genus_hparams.json` - GNN model up to genus level
- `gnn_species_hparams.json` - GNN model up to species level

# Preguntes a resoldre:
## Donat que tenim un problema de balanceig, podem solucionar-ho amb data augmentation? comparem diferents mètodes amb no augmentar les dades.

## evaluar quin encoding dels que proposem va millor per cada model

## comparar la performance dels diferents models

## provar amb totes les classes a nivell d'ordre



