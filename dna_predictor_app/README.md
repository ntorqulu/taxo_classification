## DNA Predictor App

Web based tool to classsify DNA sequences into taxonomic groups using pre-trained deep learning models.

### 1. How it works

The app uses trained neural network models to predict the taxonomic classification of DNA sequences. It supports:

#### 1.1. Multiple Model Architectures:
- CNN Models (nanni_cnn1, nanni_cnn2, CNNModel)
- MLP Model (EnhancedMLP)
- Attention-based models (nanni_att)
- BERT model for DNA (BERTTaxoModel)

#### 1.2. Multiple Encoding Methods:
- 4-row encoding
- k-mer encoding (kmer_3, kmer_4, etc.)
- Bit encoding (bits_2, bits_3, etc.)

#### 1.3. Different Taxonomic Levels:
- Order
- Family
- Class
- Phylum
- Kingdom
- ... i don't remember more


### 2. Running the app

#### Prerequisites
- Python 3.8+
- PyTorch
- Flask
- NumPy
- torch
- pandas
- scikit-learn
- matplotlib
- seaborn
- Model checkpoints in the correct format in Results folder (will only retrive the .best.pt file)

### 3. Execute it with Python

```bash
cd taxo_classification
pip install -r requirements.txt
cd dna_predictor_app
python app.py
```

- Place trained model checkpoints in the Results directory
- Each model should be in its own directory
- Checkpoint files should be named with *_best.pt suffix

### 4. Access the app
- Open your browser and go to: http://127.0.0.1:5000
- The app will also be available on your local network at the IP shown in the terminal