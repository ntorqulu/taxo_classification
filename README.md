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

### Testing the architechtures
#### Hyperparams JSON
To parse the different hyperparameters and the different options for each experiment a json can be used. When running the model you can parse the json with the option --config

```
$ PYTHONPATH=$(pwd)/src/ python src/models/main.py --config src/models/hyperparams/kmer_hparams.json --model_type basic
```

Such JSON can have the following keys:

- `batch_size`: int, size of the batch for the training process. See [DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- `bits`: int, number of bits used in the one-hot-encoding codifications of sequences. If bits = 0, then the 4xN matrix codification is used. See DNA codification section for further information. If bits is not None, then k must not be specified in the json.
- `dropout`: Float, probability of an element to be zeroed. See [p from Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout). Can be used in the cnn and enhanced_mlp models but those from Nanni 2024, the Dropout is fixed.
- `epochs`: int, number of epochs to use in the training phase.
- `eval_frequency`: int, How often to run full evaluation (epochs).
- `every_n_epochs`: int, When scheduler is done by steps, using the [torch.optim.lr_scheduler.LambdaLR "by_steps"](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR), how many epochs has to pass to the LR to be updated.
- `fast_mode`: Bool, Use faster evaluation with minimal metrics. This option was meant to optimize the running time for training but there is not much difference.
- `fc_sizes`: list, List of fully connected layer sizes for the cnn model.
- `from_checkpoint`: Bool, whether the model has to run from a checkpoint, True, or not, False. If True, from_checkpoint_path must not be None. False by default.
- `from_checkpoint_path`: Str, the path from where the checkpoint in which the training must continue.
- `hidden_size`: list or int, hidden size of the Fully connected layers for the different models. 
- `k`: int, size of the window to perform the kmerization. See DNA codification section for further information. If k is not None, then bits must not be specified in the json.
- `kernel_sizes`: int, size of the kernel for the cnn model. For the Nanni 2024 models the kernel size is fixed
- `label_column_name`: str, name of the column for wich the labels for prediction are taken. Options are: "kingdom_name", "phylum_name", "class_name" and "order_name"
- `learning_rate`: float, learning rate used for the optimizer. by default is 0.001 for the Adam and 0.01 for the SGD.
- `max_rows`: float or int, if float, values between 0 and 1, proportion of sequences used for the experiment. If int, total max number of sequences to be used in the experiment
- `min_cardinality_filters`: dict[str, int], dictionary with rank columns as keys and *minimum* cardinality as values.
Only labels in each column with cardinality higher than the specified value will be used.
- `model_type`: str, model name. Options: "basic" (default), "enhanced_mlp", "cnn", "nanni_cnn1", "nanni_cnn2" and "nanni_att"
- `momentum`: float, momentum value for the SDG optimizer.
- `num_filters`: int, List of filter counts for conv layers for the "cnn" model
- `optimizer`: str, optimizer to be used. Options are "[adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)" (default) "[sgd](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)"
- `patience`: int, Early stopping patience (epochs with no improvement). Default 5.
- `scheduler`: str, lr scheduler if use_scheduler set to True. Options are "[plateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#reducelronplateau)" (default), "[cosine](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)" or "[by_steps](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR)"
- `scheduler_patience`: int, patience used by the [ReduceLROnPlateau "plateau"](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#reducelronplateau) scheduler. Default 3
- `seed`: int, seed for reproducibity. Default 42
- `seq_len_filter`: int, sequence filter by length. Be aware that the most frequent length for the dataset is 313
- `sequence_length`: int, Input sequence length for the Nanni 2024 models. Be aware that the most frequent length for the dataset is 313. 
- `use_batch_norm`: Bool, Whether to use or not batch normalization for the enhanced_mlp model
- `use_scheduler`: Bool, Whether to use or not the scheduler for the lr
- `weight_decay`: float, weight_decay for the optimizers. For adam is 0 by default and for sgd is 1e-4 by default


# Preguntes a resoldre:
## Donat que tenim un problema de balanceig, podem solucionar-ho amb data augmentation? comparem diferents mètodes amb no augmentar les dades.

## evaluar quin encoding dels que proposem va millor per cada model

## comparar la performance dels diferents models

## provar amb totes les classes a nivell d'ordre



