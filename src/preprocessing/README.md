*PREPROCESSING PIPELINE*

***OVERVIEW***

The preprocessing package provides a pipeline for preparing DNA sequence data for taxonomic classification tasks. The pipeline consists of three main modules, each handling a specific stage of data preparation:

- formatter.py: Initial processing and taxonomic annotation of raw DNA sequence data
- cleaner.py: Quality control and cleaning of formatted sequence data
- filter.py: Fine-tuning and balancing of cleaned data for specific training scenarios
These modules are designed to be used sequentially, with formatter.py and cleaner.py typically run once as they perform time-consuming but essential operations, while filter.py can be called multiple times by the DataLoader to create custom datasets for different training scenarios.

***MODULE DETAILS***
1. formatter.py
The SequenceFormatter class handles the initial processing of raw DNA sequence data, including:

- COI Region Selection: Uses a Perl script to extract the Cytochrome c Oxidase subunit I (COI) barcode region
- Taxonomic Annotation: Builds complete taxonomic lineages using taxonomy data
- Rank Approximation: Approximates missing taxonomic ranks using related inferior ranks

Key Functions:
- select_coi_region: Extracts COI regions using specific primers
- load_taxonomy_data: Loads data from NCBI taxonomy dump files
- get_lineage_with_approximation: Creates complete taxonomic lineages with approximation
- format_data: Main pipeline function that processes raw data into a formatted dataset

Example Usage:
'''
from src.preprocessing.formatter import SequenceFormatter

formatter = SequenceFormatter()
formatted_df = formatter.format_data(
    run_coi_selection=True,
    perl_script_path="src/preprocessing/select_region.pl",
    input_tsv="data/raw/sequences.tsv",
    names_dmp="data/raw/names.dmp",
    nodes_dmp="data/raw/nodes.dmp"
)
'''

2. cleaner.py
The TaxonomyDataCleaner class provides quality control and filtering operations for sequence data:

- Sequence Quality Checks: Filters sequences based on length, N content, and non-standard bases
- GC Content Analysis: Removes sequences with outlier GC content
- Taxonomic Consistency: Checks and filters inconsistent taxonomic assignments
- Duplicate Removal: Identifies and removes duplicate sequences

Key Functions:
- filter_by_sequence_length: Removes sequences shorter than specified length
- filter_by_n_content: Removes sequences with high ambiguous base content
- filter_by_gc_content: Removes sequences with outlier GC content
require_complete_ranks: Ensures taxonomic completeness up to a specific rank
- check_taxonomy_consistency: Identifies inconsistent taxonomic assignments
- clean_data: Main pipeline function that performs all cleaning operations

Example Usage:
'''
from src.preprocessing.cleaner import TaxonomyDataCleaner

cleaner = TaxonomyDataCleaner()
cleaned_df = cleaner.clean_data(
    df=formatted_df,
    min_seq_length=299,
    max_n_percent=0.0,
    require_complete_ranks_up_to="order",
    remove_duplicates=True,
    filter_nonstandard_bases=True,
    enforce_taxonomy_consistency=True,
    filter_gc_outliers=True
)
'''

3. filter.py
The TaxonomyDataFilter class enables customized filtering and balancing of data for specific training scenarios:

- Approximation Handling: Normalizes or removes approximated taxonomic names
- Taxonomic Scope: Filters to specific taxonomic groups
- Class Balancing: Addresses class imbalance by setting minimum and maximum examples per class
- Balanced Dataset Creation: Pipelines for preparing balanced training data

Key Functions:
- clean_approximated_names: Normalizes or removes records with approximated names
- filter_small_classes: Removes classes with insufficient examples
- filter_by_taxonomy: Includes or excludes specific taxonomic groups
balance_class_representation: Balances class distribution by setting min/max examples
- filter_for_balanced_training: Main pipeline function that creates balanced datasets

Example Usage:
'''
from src.preprocessing.filter import TaxonomyDataFilter

data_filter = TaxonomyDataFilter()
balanced_df = data_filter.filter_for_balanced_training(
    input_file="data/processed/cleaned_sequences.csv",
    target_rank="genus",
    min_examples=10,
    max_examples=100,
    filter_approximated=True,
    taxonomic_scope={"phylum_name": ["Chordata"]},
    balance_sampling_strategy="diverse"
)
'''

***Workflow***
The typical workflow progresses through these stages:

1. Raw Data → Formatted Data:

- Load raw sequence data with taxonomic IDs
- Extract COI barcode region using primer sequences
- Build complete taxonomic lineages with approximation

2. Formatted Data → Cleaned Data:

- Apply quality control filters
- Remove inconsistent or incomplete records
- Ensure taxonomic consistency

3. Cleaned Data → Training Data:

- Select specific taxonomic scope
- Handle class imbalance
- Create balanced datasets for specific training tasks

***Usage Notes***
- formatter.py and cleaner.py should be run once to create a clean, high-quality dataset
- filter.py can be called multiple times with different parameters to create specialized training datasets
- For large datasets, the formatting and cleaning steps may be time-consuming
- The filter module provides flexibility to create datasets for different taxonomic levels and training tasks

***Example of Complete Pipeline***
'''
import logging
logging.basicConfig(level=logging.INFO)

# 1. Format raw data
from src.preprocessing.formatter import SequenceFormatter
formatter = SequenceFormatter()
formatted_df = formatter.format_data(run_coi_selection=True)

# 2. Clean the formatted data
from src.preprocessing.cleaner import TaxonomyDataCleaner
cleaner = TaxonomyDataCleaner()
cleaned_df = cleaner.clean_data(formatted_df)
cleaner.save_cleaned_data(cleaned_df)

# 3. Create balanced training datasets for different tasks
from src.preprocessing.filter import TaxonomyDataFilter
data_filter = TaxonomyDataFilter()

# Dataset for genus classification
genus_df = data_filter.filter_for_balanced_training(
    input_file="data/processed/cleaned_sequences.csv",
    target_rank="genus",
    min_examples=5,
    max_examples=50
)

# Dataset for family classification
family_df = data_filter.filter_for_balanced_training(
    input_file="data/processed/cleaned_sequences.csv",
    target_rank="family",
    min_examples=10,
    max_examples=100
)
'''

