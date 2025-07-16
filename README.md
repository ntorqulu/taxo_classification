# Deep Learning Models for Single-Level and Multi-Level Taxonomic Classification

## Introduction
With the advent of the new DNA sequencing technologies, methodologies based on such thecniques have unveiled a bast field of applications to infer the Biodiversity of the ecosystems to monitorize such environments and their habitats. Among these thecniques DNA metabarcoding has emerged as a powerfull tool for biomonitoring. This thechnique relies on sequencing a targuet region of the genome of thousands of individuals at the same time and their latter assignment to the different taxa. However, one of the main constrains of the method is this taxonomic assignment due to the missing information in the databases and the amount of resources required to compute such analysis. While the latter problem has been assessed by using bigger computing services and optimizing the algorithms, to be able to run such models in the field withouth internet connection is still a challenge. 

Here we propose Deep Learning models as a potential solution for such problem. While the training process is resource demanding, deppending on the architecture of the module, the predictions can be potentially manageable. In this project we explore the following bottlenecks of the implementation of this thecnology in the field of Biomonitoring.

<img width="293" height="638" alt="imatge" src="https://github.com/user-attachments/assets/c8b3461f-81f8-49d0-8aa4-966059af2b74" />

---
## Table of Content
TODO
- [Project Structure](#1-project-structure)
- [Installation](#2-installation)
- [Data Acquisition](#3-data-acquisition)
  - [Source Databases](#31-source-databases)
- [Data Preprocessing](#4-data-preprocessing)
  - [Format Raw Data](#41-format-raw-data-with-sequenceformatter-class)
- [Models](#models)
  - [Single-Rank Models](#single-rank-models)
  - [Multi-Rank Models](#multi-rank-models)

---

---
## 1. Project Structure
TODO
```bash
taxo_classification/
├── README.md
├── Dockerfile
├── run_singlerank_experiment.py
├── run_multirank_experiment.py
├── parquets/
│   ├── filtered_ranks/
│   │   ├── dataset_4rowmatrix.parquet
│   │   ├── dataset_bits_1.parquet
│   │   ├── dataset_bits_2.parquet
│   │   ├── dataset_bits_4.parquet
│   │   ├── dataset_kmer_1.parquet
│   │   ├── dataset_kmer_2.parquet
│   │   ├── dataset_kmer_3.parquet
│   │   ├── dataset_kmer_4.parquet
│   │   └── dataset_kmer_5.parquet
│   └── all_ranks/
│       ├── dataset_4rowmatrix.parquet
│       ├── dataset_bits_1.parquet
│       ├── dataset_bits_2.parquet
│       ├── dataset_bits_4.parquet
│       ├── dataset_kmer_1.parquet
│       ├── dataset_kmer_2.parquet
│       ├── dataset_kmer_3.parquet
│       ├── dataset_kmer_4.parquet
│       └── dataset_kmer_5.parquet
├── src/
│   ├── constants/
│   │   └── taxonomy_labels.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── cached_dataframe.py
│   │   ├── hierarchical_dataset.py
│   │   ├── taxo_dataloaders.py
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── main_singlerank.py
│   │   ├── main_multirank.py
│   │   ├── architectures/
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py
│   │   │   ├── bert_model.py
│   │   │   ├── cnn_model.py
│   │   │   ├── mlp_model.py
│   │   │   ├── enhanced_mlp.py
│   │   │   ├── nanni2024.py
│   │   │   ├── hierarchical_model.py
│   │   │   ├── cascade_hierarchical_model.py
│   │   │   └── gnn_hierarchical_model.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── singlerank_trainer.py
│   │   │   ├── multirank_trainer.py
│   │   │   └── sequential_trainer.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── model_factory.py
│   │   ├── hyperparams/
│   │   │   ├── singlerank/
│   │   │   │   ├── bert_hparams.json
│   │   │   │   ├── kmer_hparams.json
│   │   │   │   ├── one_hot_hparams.json
│   │   │   │   └── nanni_cnn1_hparams.json
│   │   │   └── multirank/
│   │   │       ├── hierarchical_hparams.json
│   │   │       ├── cascade_hparams.json
│   │   │       └── gnn_hierarchical_hparams.json
│   └── utils/
│       ├── __init__.py
│       └── sequence_formatter.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── checkpoints/
│   ├── singlerank/
│   └── multirank/
└── runs/
    ├── singlerank/
    └── multirank/
 ```
    
---

---

## 2. Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/taxo_classification.git
    cd taxo_classification
    ```

2. **Install dependencies:**
    ```bash
    pip install -r src/requirements.txt
    ```

3. **(Optional) Docker setup:**
    ```bash
    docker build -t taxo-classifier .
    ```

---
## 3. Data Acquisition

### 3.1. Source Databases

To obtain the database we used the sequences from [mkCOInr](https://github.com/meglecz/mkCOInr) as described in [NJORDR-MJOLNIR3](https://github.com/adriantich/NJORDR-MJOLNIR3) and can be downloaded from google drive: [NJORDR_sequences](https://drive.google.com/file/d/1YU_jIRIm9rpEC4okD5xh2qr3EnGBg3i8/view?usp=sharing), [names.dmp](https://drive.google.com/file/d/1WrRHX5Mf23ijg03K5YNaAx3dtzgIX5Zn/view?usp=sharing) and [nodes.dmp](https://drive.google.com/file/d/1D4g7PP-mdP9xqsxM9ZC_Bz9wqANkf6UN/view?usp=sharing). These are sequences from the public databases NCBI and BOLD and sequences obtained by scientific groups in University of Barcelona, Center for Advanced Studies of Blanes and Alfred Wagener Institute.

---

## 4. Data Preprocessing

To clean and format the database, the following steps were performed:

#### 4.1. Format Raw Data with `SequenceFormatter` class
- Merge the taxonomic information and cut the region defined by the Leray-XT primers using the `SequenceFormatter` class.
- For each sequence, retain only the information regarding the following ranks: **Superkingdom**, **Kingdom**, **Phylum**, **Order**, **Species**.
- If any sequence has one of these ranks empty, retrieve the information from the rank immediately below (or two levels below) and mark it as *predicted* for further standardization or removal.
- Capitalize all bases in the sequence.

#### 4.2. Clean Data with `TaxonomyDataCleaner` class
- Remove sequences shorter than **299 bp**.
- Filter out sequences with ambiguous bases (e.g., **N**).
- Filter out sequences with non-standard bases.
- Enforce taxonomy consistency across ranks: if two sequences are identical but their taxonomy assignments differ (excluding blanks), remove them as inconclusive.
- Remove duplicate sequences.
- Ensure complete taxonomic information up to the **species** level.

#### 4.3. Create Hierarchical Filtered Dataset with `TaxonomyDataFilter` class
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

#### 4.4. Create Hierarchical Dataset with `TaxonomyDataFilter` class
- Filter sequences longer than **320 bp**.
- Clean approximated or uncertain taxonomic names, removing or standardizing them.
- Does not created nested classification levels, keeps the names of the labels for each rank as it is.
- All ranks from Kingdom to Species

To replicate the creation of the database (both filtered and not filtered), run the following command from a Unix terminal:

```bash
cd taxo_classification
python -m src.preprocessing.filter
```

#### 4.5. Generating Parquet Files

**Code Reference:**  
- [`generate_parquets.py`](generate_parquets.py)
- [`src/dataset/parquet_builder.py`](src/dataset/parquet_builder.py) (`ParquetBuilder` class)

The project uses two main datasets:

| Dataset         | Source CSV                                         | Output Parquet Directory         | Filtering Criteria                        |
|-----------------|----------------------------------------------------|----------------------------------|-------------------------------------------|
| filtered_ranks  | data/parquets/filtered_ranks/hierarchical_dataset_cleaned.csv.gz        | data/parquets/filtered_ranks/    | Complete taxonomy up to `order`           |
| all_ranks       | data/parquets/all_ranks/hierarchical_dataset_cleaned.csv.gz | data/parquets/all_ranks/         | All samples, all taxonomic levels         |

**To generate all parquets for the default dataset:**
```bash
python generate_parquets.py --coding all
```
**To generate a specific encoding:**
```bash
python generate_parquets.py --coding kmer
python generate_parquets.py --coding bit
python generate_parquets.py --coding 4row
```
**To use `all_ranks`, specify the correct CSV in your script or config.**

---
### 4.4 DNA Codification

**Code Reference:**  
- [`src/feature_extraction/main.py`](src/feature_extraction/main.py) (`SequenceCoder` class)
- [`src/dataset/parquet_builder.py`](src/dataset/parquet_builder.py)

Three main approaches are implemented:

#### 4.4.1 K-merisation

- **Description:** Breaks DNA into overlapping substrings of length *k* (k-mers).
- **Implementation:** `SequenceCoder.coding_kmer_optimized`
- **Parquet Output:** `dataset_kmer_*.parquet`

#### 4.4.2 4×N Matrix (4-row Matrix)

- **Description:** Each sequence is a 4×N binary matrix (A, C, G, T × sequence length).
- **Implementation:** `SequenceCoder.coding_one_hot_4rowMatrix_optimized`
- **Parquet Output:** `dataset_4rowmatrix.parquet`

#### 4.4.3 One-hot Encoding (Bit Encoding)

- **Description:** Each nucleotide is a one-hot vector, sequence is flattened.
- **Implementation:** `SequenceCoder.coding_one_hot_bit_optimized`
- **Parquet Output:** `dataset_bits_*.parquet`

| Method         | Description                        | Output Shape      | Typical Use Case         |
|----------------|------------------------------------|-------------------|--------------------------|
| K-merisation   | Sequence split into k-mers         | (num_kmers,)      | Embeddings, RNNs, CNNs   |
| 4×N Matrix     | 4 rows (A,C,G,T) × sequence length | (4, N)            | CNNs, sequence models    |
| One-hot/Bit    | Flat vector, 4 bits per nucleotide | (4×N,)            | MLPs, simple classifiers |

<img width="1333" height="765" alt="imatge" src="https://github.com/user-attachments/assets/705a72cc-8cbd-40b0-a9a2-402bc128d4a1" />


---
## 5. Model Architectures

**Code Reference:**  
- [`src/models/architectures/`](src/models/architectures/)

**Model selection is controlled via the `model_type` parameter in the configuration JSON files.**

Supported architectures:

### 5.1 Enhanced Multi-Layer Perceptron (MLP)
**Overview**  
The Enhanced Multi-Layer Perceptron (Enhaced MLP) is a more complex version of the standard fully connected neural network, designed for taxonomic classification.

**Architecture Details**  
The Enhanced MLP is implemented in [`enhanced_mlp.py`](src/models/architectures/enhanced_mlp.py) and features:

```
class EnhancedMLP(BaseModel):
    def __init__(self, 
                input_size: int, 
                hidden_sizes: List[int],
                output_size: int, 
                dropout: float = 0.2,
                use_batch_norm: bool = True):
```

**Key Components**  
1. Configurable Hidden Layers: Variable number of hidden layers with customizable sizes.
2. Batch normalization: Applied after each linear layer to stabilize training.
3. ReLU Activation: Non-linear activation function between layers.
4. Dropout Regularization: Prevents overfitting during training.
5. Flexible Input Handling: Adapts to different input dimensions.

**Layer Architecture**  
<img width="377" height="606" alt="Screenshot 2025-07-16 at 01 23 42" src="https://github.com/user-attachments/assets/684f1b93-47ba-4ae6-b928-0c9600a42b70" />

**DNA Encoding Adaptation**  
The Enhanced MLP is desinged to handle the three DNA encoding methods implemented in this project:
- kmer encoding
- one-hot/bit encoding
- 4 row matrix encoding

**Dynamic Input Size**  
It has the hability to dynamically adapt to different input sizes in order to handle the different encodings.

```
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Handle 4-row encoding input - flatten if needed
    if x.dim() > 2:
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, features]
    
    # Handle input size mismatch dynamically
    if x.size(1) != self.input_size:
        # Rebuild the first layer with correct input size
        actual_input_size = x.size(1)
        # ... rebuild network with correct dimensions
```

This way, there is no need to specify the exact input dimension before training.

**Training Parameters**  
It has been trained with the following architecture parameters:

```
"hidden_sizes": [512, 256, 128],
"dropout": 0.3,
"use_batch_norm": true
```
  
### 5.2 Nanni CNN Variants
- **File:** [`nanni2024.py`](src/models/architectures/nanni2024.py)
- **Description:** Advanced CNNs inspired by Nanni et al. (2020, 2024), designed for DNA sequence classification.
- **Variants:**
  - **Nanni CNN1:** Compact CNN for efficient modeling.
  - **Nanni CNN2:** Deeper CNN with more filters and layers

  - **Nanni Attention Models:** Hybrid self-attention and BiLSTM for interpretability and accuracy.
- **Features:** Fixed kernel sizes, dropout, and attention layers (for attention models).
- **Configuration:** Configurable via JSON files in [`src/models/hyperparams/singlerank/`](src/models/hyperparams/singlerank/).
**Layer Architectures**  

<img width="300" height="563" alt="Nanni Attention Model" src="https://github.com/user-attachments/assets/227c3f3b-dbd8-4a59-a56a-037deb695659" />
<img width="1256" height="708"  alt="CNN1 and CNN2" src="https://github.com/user-attachments/assets/c0d63026-c6e4-4153-b783-3651dcdde3f1" />

#### 5.2.1 Nanni CNN1
**Overview:**
This is the simplest model of the ones inspired by Nanni et al. (2024). The code was addapted from Matlab to pytorch and the parameters were keps as much as possible as in the original MS.

**Architecture Details**
The nanni_cnn1 model is defined in [nanni2024.py](https://github.com/ntorqulu/taxo_classification/blob/main/src/models/architectures/nanni2024.py) and defined with:
```
class nanni_cnn1(BaseModel):
    def __init__(self,
                 sequence_length: int,
                 hidden_size: int,
                 output_size: int,
                 name: str = "nanni_cnn1"):
```
**Key Components**  
1. Convolutional layer with 16 filters, kernel size 3 and padding = "same".
2. Batch normalization 2D
3. Dropout set at 0.5 of probability
4. Fully connected NN with two layers. The hidden size was set to 8 as in the original MS

**Training**
Training was performed using the same parameters as in Nanni et al. (2024) but results clearly improved specially when increasing the hidden size of the Fully connected NN.

#### 5.2.2 Nanni CNN2
**Overview:**
This model is an improvement of the previous model. This model adds a new convolutional layer and increase the complexity of the fully connected layer by adding an extra layer and increasing the hidden size.

**Architecture Details**
The nanni_cnn2 model is defined in [nanni2024.py](https://github.com/ntorqulu/taxo_classification/blob/main/src/models/architectures/nanni2024.py) and defined with:
```
class nanni_cnn2(BaseModel):
    def __init__(self,
                 sequence_length: int,
                 output_size: int,
                 hidden_size: int = 1024,
                 name: str = "nanni_cnn2"):
```
**Key Components**  
1. Convolutional layer with 16 filters, kernel size 5 and padding = "same".
2. ReLu activation layer.
3. Convolutional layer with 36 filters, kernel size 5 and padding = "same".
4. ReLu activation layer.
5. Max pooling layer with kernel size 2
8. Dropout set at 0.5 of probability
9. ReLu activation layer
10. Fully connected NN with three layers and ReLu activation layers in between. The hidden size was set to 1024 as in the original MS

#### 5.2.3 Nanni attention
**Overview:**
This model combines multi-head self-attention, Bi-LSTMs and learnable attention pooling mecanism to generate.  

**Architecture Details**
The `nanni_att` is defined in [nanni2024.py](https://github.com/ntorqulu/taxo_classification/blob/main/src/models/architectures/nanni2024.py) and defined with:
```
class nanni_att(BaseModel):
    def __init__(
        self,
        sequence_length: int,
        output_size: int,
        num_heads: int = 8,
        embed_dim: int = 64,
        hidden_size: int = 100,
        batch_size: int = 30,
        name: str = "nanni_att",
    ):
```
**Key Components**  
1. **Input projection**: converts sequences into fixed length embeddings (64).
2. **Multi-Head Self-Attention**: allows each position of a sequence to see all the other positions to see what it must attend to. We compute it with 8 heads. Higlight global information.
3. **BiLSTM**: It gives context to the order and the directionality. We compute it with `hidden_size == 100`.
4. **Learnable attention pooling** Produces a single sequence-level representation of attention via softmax.
5. **Batch normalization**: Normalizes the pooled output.
6. **FC**: FC of `hidden_size=100` to output size.

### 5.3 BERT-based Model
**Overview:**  
The BERT-based model is a transformer-based architecture adapted to DNA sequence classification. Inspired by the Bidirectional Encoder Representations from Transformers (BERT) architecture, this model use self-attention mechanisms to capture contextual relations in DNA sequences.

**Architecture Details:**  
The BERT model is implemented in [`bert_model.py`](src/models/architectures/bert_model.py) and defined with:

```
class BERTTaxoModel(BaseModel):
    def __init__(self, 
                 vocab_size: int = 4,  # A, T, G, C
                 max_length: int = 313,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 output_size: Optional[int] = None,
                 classifier_hidden_size: int = 128,
                 name: str = "BERTTaxoModel"):
```

**Key Components**  
1. Token Embedding: Converts DNA nucleotides (A, T, G, C) to dense vector representations.
2. Positional Encoding: Adds sinusoidal position information to preserve sequence order.
3. Multi-Head Self-Attention: Captures relationships between different positions in the sequence. Can relate nucleotides across the entire sequence, making each position attend to all other positions. This way, it can learn large sequence patterns.
4. Transformer Encoder Layers: Stack of transformer blocks for deep feature extraction.
5. Layer Normalization: Applied before and after attention and feed-forward layers.
6. Classification Head: Multi-layer preceptron for final taxonomy prediction.
7. Dropout Regularization: Prevents overfitting throughout the network.

**Layer Architecture**  
<img width="1256" height="708" alt="BERT-based Model Architecture" src="https://github.com/user-attachments/assets/fde35331-1c4d-4270-b7b8-77c228c887ad" />

**DNA Encoding Adaptation**  
The model is specifically designed for 4 row matrix encoding only:
- Input Format: 2D matrix (4, seq_length)
- Processing: Converts the 4-row matrix to token IDs using argmax operation
- Character Mapping: {'A': 0, 'T': 1, 'G': 2, 'C': 3}
- Only trains with sequences of fixed length size 313 bps

**Training Parameters**  
It has been trained with the following architecture parameters:

```
{
  "vocab_size": 4,                       // DNA nucleotide vocabulary size
  "max_length": 313,                     // Maximum sequence length
  "hidden_size": 128,                    // Hidden dimension size
  "num_layers": 3,                       // Number of transformer layers
  "num_heads": 4,                        // Number of attention heads
  "dropout": 0.2,                        // Dropout probability
  "classifier_hidden_size": 128,         // Classification head hidden size
}
```


### 5.4 Connected Models
**Overview**  
The connected models are an implementation of different connected models such as Iterative
fixed input network, densely connected network (DenseNet), residual network (ResNet) or 
recurrent refinement networks

**Architecture Details**  
The connected models are implemented in [`connected_model.py`](src/models/architectures/connected_model.py) and features:

```
class ConnectedModel(BaseModel):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 models: list[nn.Module],
                 dropout: float = 0.5,
                 connected_type: str  = 'iterative',
                 connected_models: dict[str, str] = None):
```

**Model types**

Iterative Fixed Input (`iterative`)   
 
    x = net1(x0)
    x1 = net2(concat(x, x0))
    x2 = net3(concat(x1, x0))
    x3 = net4(concat(x2, x0))

Densely Connected Network or DenseNet (`densenet`)
 
    x1 = net1(x0)
    x2 = net2(concat(x0, x1))
    x3 = net3(concat(x0, x1, x2))

Residual Network or ResNet (`resnet`)
  
    x1 = net1(x0) + x0
    x2 = net2(x1) + x1
    x3 = net3(x2) + x2

Recurrent refinement (`recurrent`). Note that recurrent is the same as iterative, but the network remains the same type at each step.":

    x = net(x0)
    x1 = net(concat(x, x0))
    x2 = net(concat(x1, x0))
    x3 = net(concat(x2, x0))

**Configuration:**

Configurable via JSON files in [`src/models/hyperparams/singlerank/`](src/models/hyperparams/singlerank/). For examples, 
you can check `connected_hparams.json` and the parameters `model_type`, `connected_type`, and `connected_models`.


**Layer Architectures**  
<img width="1256" height="708" alt="Connected Models Architectures" src="https://github.com/user-attachments/assets/f39aed16-cf3a-47d9-8707-0b61ce4e958b" />


---

## 6. Configuration and Hyperparameters

**Code Reference:**  

- [`src/models/hyperparams/`](src/models/hyperparams/)
- [`src/models/main_singlerank.py`](src/models/main_singlerank.py)
- [`src/models/main_multirank.py`](src/models/main_multirank.py)

#### Hyperparams JSON
To parse the different hyperparameters and the different options for each experiment a json can be used. When running the model you can parse the json with the option --config

```
$ PYTHONPATH=$(pwd)/src/ python src/models/main.py --config src/models/hyperparams/kmer_hparams.json --model_type basic
```

Such JSON can have the following keys:

- `batch_size`: int, size of the batch for the training process. See [DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- `bits`: int, number of bits used in the one-hot-encoding codifications of sequences. If bits = 0, then the 4xN matrix codification is used. See DNA codification section for further information. If bits is not None, then k must not be specified in the json.
- `connected_type`: str. Only for `model_type="connected"`. Specifies the type of connected model. Valid values are "iterative", "densenet", "resnet", and "recurrent".
- `connected_models`: dict[str, str]. Only for `model_type="connected"`. Specifies the model to be used at each level. Each value must be one of the valid `model_type` options (currently only tested with "nanni_cnn2").
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
Only labels in each column with cardinality higher than the specified value will be used. In case of multiple labels, the order of the labels will be taken into account.
- `model_type`: str, model name. Options: "basic" (default), "enhanced_mlp", "cnn", "nanni_cnn1", "nanni_cnn2" 
, "nanni_att" and "connected"
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

## 7. Training

**Code Reference:**  
- [`src/models/main_singlerank.py`](src/models/main_singlerank.py)
- [`src/models/main_multirank.py`](src/models/main_multirank.py)
- [`src/models/training/`](src/models/training/)

This project supports two main types of classification models: **Singlerank** (single taxonomic level) and **Multirank** (hierarchical, multi-level). Both are configured via JSON files and run from the command line.

### 7.1. Single-rank Models

**Run command:**
```bash
PYTHONPATH=$(pwd)/src/ python src/models/main_singlerank.py --config src/models/hyperparams/singlerank/<your_config>.json
```

**Example:**
  ```bash
  PYTHONPATH=$(pwd)/src/ python src/models/main_singlerank.py --config src/models/hyperparams/singlerank/kmer_hparams.json
  ```

**Available model types:**
- `basic` (MLP)
- `enhanced_mlp`
- `cnn`
- `nanni_cnn1`
- `nanni_cnn2`
- `nanni_att`
- `nanni_att_kmer`
- `bert`

**Key config parameters (singlerank):**

| Parameter              | Type        | Description |
|------------------------|-------------|-------------|
| parquets_path          | str         | Path to data parquets (default: auto) |
| dataset_name           | str         | Dataset folder name (e.g. filtered_ranks, all_ranks) |
| max_rows               | float/int   | Proportion (0-1) or max number of rows |
| seed                   | int         | Random seed |
| epochs                 | int         | Number of training epochs |
| batch_size             | int         | Batch size |
| learning_rate          | float       | Learning rate |
| weight_decay           | float       | Weight decay for optimizer |
| label_column_name      | str         | Target label column (e.g. order_name) |
| k                      | int/null    | k-mer size (if using k-mer encoding) |
| bits                   | int/null    | One-hot encoding bits (if using one-hot/4xN) |
| model_type             | str         | Model architecture (see above) |
| balancing_method       | str         | 'none', 'loss_soft', 'loss_strong' |
| optimizer              | str         | 'adam' or 'sgd' |
| use_scheduler          | bool        | Use LR scheduler |
| scheduler              | str         | 'plateau', 'cosine', 'by_step' |
| scheduler_patience     | int         | Patience for scheduler |
| scheduler_factor       | float       | LR reduction factor |
| patience               | int         | Early stopping patience |
| hidden_size(s)         | int/list    | Hidden layer sizes (model-specific) |
| dropout                | float       | Dropout probability |
| use_batch_norm         | bool        | Use batch normalization |
| kernel_sizes           | list        | CNN kernel sizes |
| num_filters            | list        | CNN filter counts |
| fc_sizes               | list        | CNN fully connected sizes |
| fast_mode              | bool        | Fast evaluation mode |
| eval_frequency         | int         | Evaluation frequency (epochs) |
| seq_len_filter         | int         | Filter sequences by length |
| every_n_epochs         | int         | Scheduler step interval (by_step) |
| vocab_size             | int         | (BERT) Vocabulary size |
| max_length             | int         | (BERT) Max sequence length |
| num_layers             | int         | (BERT) Number of transformer layers |
| num_heads              | int         | (BERT/Attention) Number of heads |
| classifier_hidden_size | int         | (BERT) Classifier hidden size |
| embed_dim              | int         | (Attention) Embedding dimension |
| experiment_id          | str         | Optional experiment name/ID |

See `src/models/hyperparams/singlerank/` for example configs for each model type.

---

### 7.2. Multirank (Hierarchical) Models

**Run command:**
```bash
PYTHONPATH=$(pwd)/src/ python src/models/main_multirank.py --config src/models/hyperparams/multirank/<your_config>.json
```

- Example:
  ```bash
  PYTHONPATH=$(pwd)/src/ python src/models/main_multirank.py --config src/models/hyperparams/multirank/hierarchical_hparams.json
  ```

**Available model types:**
- `hierarchical`
- `cascade_hierarchical`
- `gnn_hierarchical`

**Key config parameters (multirank):**

| Parameter              | Type        | Description |
|------------------------|-------------|-------------|
| parquets_path          | str         | Path to data parquets (default: auto) |
| dataset_name           | str         | Dataset folder name (e.g. filtered_ranks, all_ranks) |
| max_rows               | float/int   | Proportion (0-1) or max number of rows |
| seed                   | int         | Random seed |
| epochs                 | int         | Number of training epochs |
| batch_size             | int         | Batch size |
| learning_rate          | float       | Learning rate |
| weight_decay           | float       | Weight decay for optimizer |
| label_column_name      | str         | Target label column (e.g. order_name, genus_name, species_name) |
| k                      | int/null    | k-mer size (if using k-mer encoding) |
| bits                   | int/null    | One-hot encoding bits (if using one-hot/4xN) |
| model_type             | str         | Model architecture (see above) |
| balancing_method       | str         | 'none', 'loss_soft', 'loss_strong' |
| optimizer              | str         | 'adam' or 'sgd' |
| use_scheduler          | bool        | Use LR scheduler |
| scheduler              | str         | 'plateau', 'cosine', 'by_step' |
| scheduler_patience     | int         | Patience for scheduler |
| scheduler_factor       | float       | LR reduction factor |
| patience               | int         | Early stopping patience |
| shared_hidden_sizes    | list        | Shared hidden layer sizes |
| level_specific_sizes   | dict        | Per-level hidden layer sizes |
| dropout                | float       | Dropout probability |
| use_confidence_weighting| bool       | (Cascade) Use confidence weighting |
| cascade_weight         | float       | (Cascade) Cascade loss weight |
| confidence_weight      | float       | (Cascade) Confidence loss weight |
| level_weights          | dict        | Per-level loss weights |
| loss_type              | str         | Loss type (e.g. 'cross_entropy') |
| focal_alpha            | float       | Focal loss alpha (if used) |
| focal_gamma            | float       | Focal loss gamma (if used) |
| gnn_layers             | int         | (GNN) Number of GNN layers |
| use_attention          | bool        | (GNN) Use attention in GNN |
| graph_weight           | float       | (GNN) Graph loss weight |
| consistency_weight     | float       | (GNN) Consistency loss weight |
| fast_mode              | bool        | Fast evaluation mode |
| eval_frequency         | int         | Evaluation frequency (epochs) |
| seq_len_filter         | int         | Filter sequences by length |
| experiment_id          | str         | Optional experiment name/ID |

See `src/models/hyperparams/multirank/` for example configs for each model type and taxonomic level.

---

**Notes:**
- All config parameters can be overridden by editing the JSON config file.
- For hierarchical models, the available taxonomic levels are detected automatically from the dataset columns.
- For more details on each parameter, see the example config files and code comments in `src/models/main_singlerank.py` and `src/models/main_multirank.py`.

---
### 9. Category Balance for training
### 10. Hyperparameter optimisation

---
## 11. Results

**Trained Models**

In the `Results` folder, you will find the trained models. In each subfolder, you'll find:

- `*_best.pt`: the PyTorch checkpoint file.
- `.json`: a file with the parameters used for training.
- `.log`: a file with the log output during training.
- `README.md`: a file with the commands used to train the model.

**Architecture comparison**

In the following charts, you can see a comparison of different architectures in terms of Accuracy, F1, and Precision on the test datasets. Some models are missing from one of the charts due to poor performance or because they haven't been trained yet. Note that the highlighted model names are the ones common to both charts


This chart corresponds to the **order level** (251 different categories in the training dataset) of the taxonomy, and it shows that the Enhanced MLP model achieves the best scores across the different metrics:.

<img width="596" height="567" alt="imatge" src="https://github.com/ntorqulu/taxo_classification/blob/main/Results/PLOTS/arch_order.png" />


The following chart corresponds to the **genus level** (2,246 different categories in the training dataset) of the taxonomy, and the best results are from the Enhanced MLP, Nanni 2024 with attention, and Nanni 2024 CNN2 models. The Nanni models performs better in the genus level that order level.

<img width="596" height="567" alt="imatge" src="https://github.com/ntorqulu/taxo_classification/blob/main/Results/PLOTS/arch_genus.png" />


**Encoding comparison**

In the next charts we'll see the differences in terms of encoding using the same metrics as in the previous architecture comparison. We have KMer encodings for K=1 (k1), K=2 (k2), K=3 (k3), and K=4 (k4); one-hot encodings for 1 (bits1), 2 (bits2), 3 (bits3), and 4 bits (bits4); and a 4-row encoding labeled as bits0.

The first chart corresponds to the order level of the taxonomy, and the second one to the genus level. As you can observe, the 4-row encoding performs better than the others, although not far from the one-hot encodings. The KMer encodings perform worse, especially those with lower K values (lower K means less information). Note that in our case, the 4-row encoding has the same size as the one-hot encoding with 4 bits, but it still performs better.

<img width="569" height="575" alt="imatge" src="https://github.com/ntorqulu/taxo_classification/blob/main/Results/PLOTS/coding_order.png" />

<img width="569" height="575" alt="imatge" src="https://github.com/ntorqulu/taxo_classification/blob/main/Results/PLOTS/coding_genus.png" />


---

## 12. DNA Prediction App

The project includes an application for taxonomic classification of DNA sequences without requiring programming knowledge. This app allows researchers to quickly classify DNA sequences using out pre-trained models.

### 12.1. App Overview

- Direct sequence input via text field
- Multiple model selection (BERT, CNN, Cascade, etc.)
- Confidence scores for each prediction level
- Prediction of the most confident label based on the sequence inputed

### 12.2. Installation

To run with python in the command line:
  ```bash
  # Install the app dependencies
  pip install -r app/requirements.txt

  # Launch the app
  python -m app.run
  ```

## 13. Experiments
This section presents the experiments conducted in this project

### 13.1 Model Inference Using the DNA Predictor App

**Hypothesis:** We expect that models trained on taxonomic DNA sequences will be able to classify new, unseen sequences to their correct taxonomic rank. We hypothesize that:
- For labels present in the training split, the model will predict this exact label
- For labels not present in the training split, the model will predict labels that are close in the taxonomic tree.
- Different encoding methods (kmer, onehot, 4row) and model architectures will vary in performance across different taxonomic ranks.

**Experiment Setup:**
- **Test Data:** Real DNA sequences from [`PCA/real_seqs.tsv`](PCA/real_seqs.tsv) not present in the training dataset
- **Models Tested:** All trained model architectures (Enhanced MLP, CNN, Nanni variants, BERT) in their best epoch with different encodings, stored in [`Results`](Results/)
- **Evaluation Tool:** [`dna_predictor_app`](dna_predictor_app/) for inference and comparison
- **Metrics:** Confidence scores, and taxonomic proximity analysis
- **Test Categories:** 
  1. Known taxa (present in training data)
  2. Unknown taxa (not in training data)
  3. Sequences with incomplete taxonomic information

**Results:**

#### 13.1.1 Perfect Classification of Known Taxa

For sequences where both order and genus labels were present in the training split, the models achieved near-perfect classification:

**Sequence:**
TTTATCAAGTAACATTGCTCATTCTGGTGCTTCAGTTGACTTATCAATTTTCTCTTTACATTTAGCGGGTGCTTCGTCAATTTTAGGTGCCATTAATTTTATGTCTACAGTTATTAACATACGAGCTGAAACACTGACATTTGATCGACTTCCATTATTTGTCTGAAGAGTATTTATTACTGTAATTCTTTTACTTTTATCACTTCCAGTACTAGCAGGAGCTATTACTATGTTGCTAACAGATCGAAATCTGAATACCTCATTTTTTGATCCAACAGGAGGTGGAGATCCAATCTTATACCAACATCTATTT

```

| Taxonomic Rank | Taxon Name | Sample Count |
|----------------|------------|--------------|
| Order          | Balanomorpha | 2,406 |
| Family         | Balanidae | 697 |
| Genus          | Amphibalanus | 80 |
```

- **Best Performing Model:** Enhanced MLP with k-mer encoding for genus classification and 4-row encoding for order classification

![Model Performance on Known Taxa](readme_files/gif/known_taxa_prediction.gif)

*Figure 1: DNA Predictor App showing perfect classification of known taxa using Enhanced MLP with optimal encoding methods*

#### 13.1.2 Taxonomically Proximate Predictions for Unknown Taxa

For genus *Elasmopus* (absent from training data), the model predictions were taxonomically coherent:

**Sequence:**
TTTAGCCTCTTCTTTAGGTCATAGAGGAAGCTCCGTGGACCTAGCAATTTTTTCTTTACATCTAGCAGGAGCTTCTTCTATCTTAGGAGCTATTAATTTCATCACTACTGTAATTAATATACGAACCGCAGGAATATACATAGACCAAATCCCCTTATTTGTTTGATCTGTTTTCATTACAGCCATTCTACTTCTGCTTTCTCTTCCTGTTCTTGCTGGAGCAATTACCATACTTCTCACTGATCGAAACCTAAATACTTCTTTCTTCGACCCTTGTGGGGGAGGTGATCCAATCCTTTACCAACATTTATTC

```

| Taxonomic Rank | Taxon Name | Sample Count |
|----------------|------------|--------------|
| Order          | Amphipoda | 4,760 |
| Family         | Maeridae | 0 (absent) |
| Genus          | Elasmopus | 0 (absent) |
```

- **Predicted Genus:** *Pontogammarus* 
- **Taxonomic Relationship:** Both genera belong to the same family (Pontogammaridae) and order (Amphipoda)
- **Biological Significance:** The prediction demonstrates the model's ability to capture phylogenetic relationships


![Elasmopus Prediction](readme_files/gif/elasmopus_prediction.gif)

*Figure 2: DNA Predictor App predicting Pontogammarus for an Elasmopus sequence*

<img width="486" height="417" alt="Screenshot 2025-07-16 at 01 52 01" src="https://github.com/user-attachments/assets/d8267493-d168-494e-aed6-4a7143ed959a" />


*Figure 3: Phylogenetic relationship between Elasmopus and Pontogammarus showing their taxonomic proximity*

#### 13.1.3 Gap-Filling for Incomplete Taxonomic Data

For sequences with incomplete taxonomic information (blank genus labels), the models provided taxonomically consistent predictions:

**Sequence:**
ATTGTCAAGAAATTTAGCTCATTCTGGGGCTGCATTAGATTGTGCTATTTTTTCACTTCATTTGGCTAGGGTTTCTAGTATTTTAAGGTCTTTAAATTTTATAACTACTTTGTTTAATATAAAAGTTAAGAGGTGAGGGATGTTCTCCATATCTCTGTTTTGTTGAACTGTATTAGTTACTACTATTTTGTTATTATTATCTTTACCTGTTTTAGCTGCAGCTATTACAATATTACTTTTCGATCGAAATTTTAATACTTCTTTTTTTGATCCCTCTGGGAGAAGAGATCCGGTTTTGTATCAGCACTTGTTT

```

| Taxonomic Rank | Taxon Name | Sample Count |
|----------------|------------|--------------|
| Order          | Stolidobranchia | 1,240 |
| Family         | Styelidae | 45 |
| Genus          | Botrylloides | 32 |
| Species        | Not known | 0 (absent) |
```

![Gap Filling Prediction](readme_files/gif/gap_filling_prediction.gif)

*Figure 4: Model predicting taxonomically consistent genus for sequences with incomplete taxonomic information*
#### 13.2 Interpretability
We performed some interpretability analysis with two different models that could explain the performance showed in the 13.1.3. 

##### 13.2.1 PCA
We used [57 sequences](https://github.com/ntorqulu/taxo_classification/blob/main/PCA/real_seqs.tsv) from a real experiment that tried to detect invasive species in the harbour of Blanes to obtain their embeddings and compare the taxonomical classification performed with the standard software and pipelines for metabarcoding analisis with the PCA representation in 2D. The embedding was obtained by converting to a vector the output of the convolutional layer the trained Nanni_cnn1 model trained at different levels. We detected that those sequences identified as being part of the same genus were clustering together. In addition the separation of two clouds preserved the evolutive information at the phylum level in which we detected that the chordata were separated from the rest of the groups. See Figure 4 and Figure 5. 

<img width="486" height="417" src="https://github.com/ntorqulu/taxo_classification/blob/main/PCA/genus.png" />

*Figure 4: PCA representation of the embeddings obtained from the convolutional layer of the nanni_cnn1 at genus level*


<img width="486" height="417" src="https://github.com/ntorqulu/taxo_classification/blob/main/PCA/phylum.png" />

*Figure 5: PCA representation of the embeddings obtained from the convolutional layer of the nanni_cnn1 at phylum level*

##### 13.2.2 Attention results

We also performed an analysis calculating the mean values of the attention for the preddicion of certain groups. As shown in the Figure 6 animation, closer groups show higher values in the same positions and this pattern is repeated as the taxonomic level increase grouping the barcodes of the animation in the same group. 

![Attention interpretation](https://github.com/ntorqulu/taxo_classification/blob/main/Results/PLOTS/ezgif-6cbd633b3cea46.gif)

*Figure 6: Attention results showing simmilarities between related groups*

**Conclusions:**

1. **Encoding Optimization:** Different taxonomic levels benefit from different encoding methods - k-mer encoding is best for genus-level classification because it has more data to train while 4-row encoding performs better for order-level classification because the encoding provides more spatial information.

2. **Phylogenetic Awareness:** Models demonstrate understanding of phylogenetic relationships, predicting proximal classes for unknown taxa. This suggests that the models learn evolutionary relationships present in the DNA sequences.

3. **Practical Application:** The models show potential for gap-filling and correction tasks in incomplete taxonomic databases.

4. **Model Architecture Insights:** Enhanced MLP consistently outperformed other architectures, likely due to its simpler architecture.

**Future Hypotheses:**
- **Taxonomic Hierarchy:** Models may perform better at higher taxonomic levels due to increased training data availability. Data augmentation or incorporation of the cascade hierarchical models would benefit the app.


---

## 13. License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## 14. References

TODO: Add more references
- Nanni, L., et al. (2020, 2024). [Deep learning architectures for DNA sequence classification.](https://www.mdpi.com/3054648)
- Arias, P. M., Sadjadi, N., Safari, M., Gong, Z., Wang, A. T., Haurum, J. B., Zarubiieva, I., Steinke, D., Kari, L., Chang, A. X., Lowe, S. C., & Taylor, G. W. (2025). [BarcodeBERT: Transformers for Biodiversity Analysis](https://arxiv.org/abs/2311.02401)
- [mkCOInr](https://github.com/meglecz/mkCOInr)
- [NJORDR-MJOLNIR3](https://github.com/adriantich/NJORDR-MJOLNIR3)

---

## 15. Open Research Questions

- How can data augmentation address class imbalance in this context?
- Which encoding method performs best for each model type?
- How do different models compare in terms of classification performance?
- What is the impact of including all classes at the order level?

---

