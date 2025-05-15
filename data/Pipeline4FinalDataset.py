# This script is the one used to create the final dataset for the training of the models. It includes the following steps:

import logging
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
        
