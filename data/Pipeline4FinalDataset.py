# This script is the one used to create the final dataset for the training of the models. It includes the following steps:

import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)

# 1. Format raw data
from src.preprocessing.formatter import SequenceFormatter
formatter = SequenceFormatter()

perl_script_path = "src/preprocessing/select_region.pl"
input_tsv = "data/raw/NJORDR_sequences.tsv"
names_dmp = "data/raw/names.dmp"
nodes_dmp = "data/raw/nodes.dmp"

formatted_df = formatter.format_data(
        run_coi_selection=True,
        perl_script_path=perl_script_path,
        input_tsv=input_tsv,
        names_dmp=names_dmp,
        nodes_dmp=nodes_dmp
    )
formatted_df.to_csv("data/processed/formatted_sequences.csv", index=False)

# 2. Clean the formatted data
from src.preprocessing.cleaner import TaxonomyDataCleaner
cleaner = TaxonomyDataCleaner()

# input_file = "data/processed/formatted_sequences.csv"

# df = pd.read_csv(input_file, dtype=str)
        
cleaned_df = cleaner.clean_data(
    formatted_df,
    min_seq_length=299,
    max_n_percent=0.0,
    require_complete_ranks_up_to="species",
    remove_duplicates=True,
    filter_nonstandard_bases=True,
    enforce_taxonomy_consistency=True,
    filter_gc_outliers=True
)

# Save the cleaned data
output_file = cleaner.save_cleaned_data(cleaned_df)

# Load the dataframe (replace 'your_file.csv' with the actual file path)
data = pd.read_csv("data/processed/cleaned_sequences.csv", dtype=str)

# Step 1: Filter by max sequence length
max_seq_length = 320
data = data[data['sequence'].str.len() <= max_seq_length]
# Step 2: Define level_1 categories
level_1 = ['Metazoa', 'Viridiplantae', 'Fungi', 'Other_euk', 'No_euk']

# Step 3: Create the level_1 column based on kingdom_name
data['level_1'] = data['kingdom_name']

# Step 4: Apply conditions to modify level_1
data.loc[data['kingdom_name'].str.contains('Bacteria', na=False), 'level_1'] = 'No_euk'
data.loc[~data['level_1'].isin(level_1), 'level_1'] = 'Other_euk'

# Repeat the process for level_2, level_3, and level_4
# Step 5: Define level_2 categories
level_2 = ['Arthropoda', 'Chordata', 'Mollusca', 'Annelida', 'Echinodermata', 
           'Platyhelminthes', 'Cnidaria', 'Other_metazoa', 'No_metazoa']

# Step 6: Create the level_2 column based on phylum_name
data['level_2'] = data['phylum_name']

# Step 7: Apply conditions to modify level_2
data.loc[~data['level_1'].str.contains('Metazoa', na=False), 'level_2'] = 'No_metazoa'
data.loc[~data['level_2'].isin(level_2), 'level_2'] = 'Other_metazoa'

# Step 8: Define level_3 categories
level_3 = ['Insecta', 'Arachnida', 'Malacostraca', 'Collembola', 'Hexanauplia', 
           'Thecostraca', 'Branchiopoda', 'Diplopoda', 'Ostracoda', 'Chilopoda', 'Pycnogonida',
           'Other_arthropoda','No_arthropoda']

# Step 9: Create the level_3 column based on class_name
data['level_3'] = data['class_name']

# Step 10: Apply conditions to modify level_3
data.loc[~data['level_2'].str.contains('Arthropoda', na=False), 'level_3'] = 'No_arthropoda'
data.loc[~data['level_3'].isin(level_3), 'level_3'] = 'Other_arthropoda'

# Step 11: Define level_4 categories
level_4 = ['Diptera', 'Lepidoptera', 'Hymenoptera', 'Coleoptera', 'Hemiptera', 
           'Trichoptera', 'Orthoptera', 'Ephemeroptera', 'Odonata', 'Blattodea', 
           'Thysanoptera', 'Psocoptera', 'Plecoptera', 'Neuroptera',
           'Other_insecta','No_insecta']

# Step 12: Create the level_4 column based on order_name
data['level_4'] = data['order_name']

# Step 13: Apply conditions to modify level_4
data.loc[~data['level_3'].str.contains('Insecta', na=False), 'level_4'] = 'No_insecta'
data.loc[~data['level_4'].isin(level_4), 'level_4'] = 'Other_insecta'

# Save the updated dataframe (optional)
data.to_csv("data/database.csv", index=False)
        
