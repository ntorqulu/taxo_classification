# BERT Model Implementation for Taxonomic Classification

## Overview

This pull request introduces a BERT-based model for taxonomic classification of DNA sequences, integrating seamlessly with the existing codebase that supports both single-rank and multi-rank classification approaches. The implementation provides a transformer-based architecture specifically designed for DNA sequence analysis using 4-row encoding.

## 🏗️ Code Structure

### Core Architecture

The BERT model is implemented in `src/models/architectures/bert_model.py` with the following key components:

- **BERTTaxoModel**: A transformer-based model inheriting from `BaseModel`
- **4-row encoding support**: Specifically designed for DNA sequence representation (A, T, G, C)
- **Configurable architecture**: Adjustable hidden size, layers, heads, and classifier dimensions
- **Positional encoding**: Fixed sinusoidal positional encodings for sequence awareness
- **Global pooling**: Mean pooling over sequence length for classification

### Key Features

1. **DNA-specific vocabulary**: 4-character vocabulary (A, T, G, C) with learnable embeddings
2. **Variable sequence length**: Handles sequences up to `max_length` with truncation
3. **Attention masking**: Proper handling of variable-length sequences
4. **Modular classifier**: Configurable classification head with multiple layers
5. **Model persistence**: Save/load functionality with configuration preservation

### Integration Points

#### Model Factory (`src/models/utils/model_factory.py`)
```python
elif model_type == "bert":
    return BERTTaxoModel(
        vocab_size=kwargs.get("vocab_size", 5),
        max_length=kwargs.get("max_length", 512),
        hidden_size=kwargs.get("hidden_size", 256),
        num_layers=kwargs.get("num_layers", 6),
        num_heads=kwargs.get("num_heads", 8),
        dropout=kwargs.get("dropout", 0.3),
        output_size=kwargs["output_size"],
        classifier_hidden_size=kwargs.get("classifier_hidden_size", 256),
        name=kwargs.get("name", "BERTTaxoModel"),
    )
```

#### Data Loading (`src/dataset/taxo_dataloaders.py`)
- Custom `bert_collate_fn` for handling 4-row encoding tensors
- Automatic detection and application of BERT-specific data loading
- Multiprocessing disabled for BERT to avoid serialization issues

## 🚀 Usage Instructions

### Single-Rank Classification

The BERT model is fully integrated into the single-rank classification pipeline.

#### 1. Configuration

Create or modify a hyperparameter file (e.g., `src/models/hyperparams/singlerank/bert_hparams.json`):

```json
{
  "parquets_path": "",
  "dataset_name": "filtered_ranks",
  "max_rows": 1.0,
  "seed": 123,
  "epochs": 30,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "weight_decay": 1e-5,
  "label_column_name": "order_name",
  "k": null,
  "bits": 0,
  
  "model_type": "bert",
  "balancing_method": "none",
  "optimizer": "adam",
  "use_scheduler": true,
  "scheduler": "cosine",
  "scheduler_patience": 3,
  "scheduler_factor": 0.5,
  "patience": 5,
  
  "vocab_size": 4,
  "max_length": 313,
  "hidden_size": 128,
  "num_layers": 3,
  "num_heads": 4,
  "dropout": 0.2,
  "classifier_hidden_size": 128,

  "experiment_id": "bert_order_classification"
}
```

#### 2. Running the Experiment

```bash
# From the project root directory
python run_singlerank_experiment.py --config src/models/hyperparams/singlerank/bert_hparams.json
```

#### 3. Available Model Types for Single-Rank

- `basic`: Simple MLP model
- `enhanced_mlp`: Enhanced MLP with batch normalization
- `cnn`: Convolutional Neural Network
- `nanni_cnn1`: Nanni et al. CNN variant 1
- `nanni_cnn2`: Nanni et al. CNN variant 2
- `nanni_att`: Nanni et al. attention model
- `bert`: BERT transformer model (new)

### Multi-Rank Classification

The multi-rank classification system supports hierarchical taxonomy classification across multiple taxonomic levels.

#### 1. Configuration

Create or modify a hyperparameter file (e.g., `src/models/hyperparams/multirank/hierarchical_hparams.json`):

```json
{
  "parquets_path": "",
  "dataset_name": "filtered_ranks",
  "max_rows": 1.0,
  "seed": 123,
  "epochs": 15,
  "batch_size": 30,
  "learning_rate": 0.001,
  "weight_decay": 1e-5,
  "label_column_name": "order_name",
  "k": 4,
  "bits": null,
  "hidden_size": null,
  
  "model_type": "hierarchical",
  "balancing_method": "none",
  "optimizer": "adam",
  "use_scheduler": true,
  "scheduler": "plateau",
  "scheduler_patience": 3,
  "scheduler_factor": 0.5,
  "patience": 5,
  
  "shared_hidden_sizes": [512, 256],
  "level_specific_sizes": {
    "kingdom_name": [128, 64],
    "phylum_name": [128, 64],
    "class_name": [128, 64],
    "order_name": [128, 64]
  },
  "dropout": 0.3,
  
  "level_weights": {
    "kingdom_name": 1.0,
    "phylum_name": 1.2,
    "class_name": 1.4,
    "order_name": 1.6
  },
  "loss_type": "cross_entropy",
  "focal_alpha": 1.0,
  "focal_gamma": 2.0,

  "fast_mode": true,
  "eval_frequency": 5,
  "seq_len_filter": 313
}
```

#### 2. Running the Experiment

```bash
# From the project root directory
python run_multirank_experiment.py --config src/models/hyperparams/multirank/hierarchical_hparams.json
```

#### 3. Available Model Types for Multi-Rank

- `hierarchical`: Basic hierarchical model
- `cascade_hierarchical`: Cascade hierarchical model with confidence weighting
- `gnn_hierarchical`: Graph Neural Network hierarchical model

#### 4. Flexible Data Path Configuration

The multi-rank script now supports flexible data path specification:

```bash
# Option 1: Command line argument
python run_multirank_experiment.py --config config.json --data_path data/parquets/filtered_ranks

# Option 2: Config file (parquets_path field)
python run_multirank_experiment.py --config config.json

# Option 3: Default path
python run_multirank_experiment.py --config config.json --dataset_name all_ranks
```

## 🔧 Technical Details

### BERT Model Architecture

```python
class BERTTaxoModel(BaseModel):
    def __init__(self, 
                 vocab_size: int = 4,  # A, T, G, C
                 max_length: int = 512,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 output_size: Optional[int] = None,
                 classifier_hidden_size: int = 128,
                 name: str = "BERTTaxoModel"):
```

**Key Components:**
- **Embedding Layer**: Converts DNA tokens to dense vectors
- **Positional Encoding**: Fixed sinusoidal encodings for sequence position
- **Transformer Encoder**: Multi-head self-attention with feed-forward networks
- **Classification Head**: Multi-layer perceptron for final classification
- **Global Pooling**: Mean pooling over sequence dimension

### Data Processing

The BERT model expects 4-row encoding format:
- Input shape: `[batch_size, 4, sequence_length]`
- Each row represents one DNA base (A, T, G, C)
- One-hot encoding converted to token IDs via argmax
- Variable-length sequences handled with attention masking

### Training Features

- **Label balancing**: Support for weighted loss functions
- **Learning rate scheduling**: Cosine annealing and plateau schedulers
- **Early stopping**: Configurable patience for overfitting prevention
- **Checkpointing**: Automatic model saving and loading
- **TensorBoard logging**: Training metrics and visualization

## 📊 Performance Considerations

### Memory Usage
- BERT models require more memory due to attention mechanisms
- Recommended batch sizes: 16-32 for typical GPU memory
- Sequence length affects memory quadratically due to attention

### Training Time
- Transformer models train slower than CNNs/MLPs
- Consider using `fast_mode` for quick experiments
- Multi-GPU training supported via PyTorch DataParallel

### Hyperparameter Tuning
- Start with smaller models (fewer layers, smaller hidden size)
- Adjust learning rate based on model size
- Monitor validation loss for early stopping

## 🧪 Testing and Validation

The implementation includes comprehensive testing:

```bash
# Run model tests
python -m pytest src/models/architectures/test_bert_model.py

# Run data loading tests
python -m pytest src/dataset/test/test_taxo_dataloaders.py

# Run integration tests
python -m pytest src/models/test/test_model_factory.py
```

## 🔄 Backward Compatibility

- All existing model types remain fully functional
- No breaking changes to existing APIs
- BERT model integrates seamlessly with existing pipelines
- Configuration files maintain backward compatibility

## 📝 Example Workflows

### Quick Start: BERT Classification

```bash
# 1. Navigate to project root
cd taxo_classification

# 2. Run BERT experiment
python run_singlerank_experiment.py --config src/models/hyperparams/singlerank/bert_hparams.json

# 3. Monitor training
tensorboard --logdir runs/
```

### Multi-Rank Experiment

```bash
# 1. Run hierarchical classification
python run_multirank_experiment.py --config src/models/hyperparams/multirank/hierarchical_hparams.json

# 2. Use different dataset
python run_multirank_experiment.py --config config.json --dataset_name all_ranks
```

## 🎯 Future Enhancements

- Support for pre-trained DNA language models
- Multi-modal fusion with additional features
- Attention visualization tools
- Distributed training support
- Model compression techniques

## 📚 References

- Vaswani, A., et al. "Attention is all you need." NeurIPS 2017
- Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019
- DNA sequence encoding techniques for deep learning

---

This implementation provides a robust, scalable foundation for transformer-based taxonomic classification while maintaining compatibility with the existing codebase architecture. 