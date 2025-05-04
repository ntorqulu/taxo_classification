import pandas as pd
import os
import logging
from tqdm import tqdm

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging for the application."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger('taxonomy_filter')

def filter_taxonomy_data(input_file, output_file=None):
    """
    Filter taxonomy data to match the R code implementation.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file with taxonomy data
    output_file : str
        Path to save the filtered data
    focus_on_insects : bool
        Whether to keep only insects or all taxonomic groups
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe with added classification columns
    """
    logger = logging.getLogger('taxonomy_filter')
    logger.info(f"Loading taxonomy data from {input_file}")
    
    # Load data
    data = pd.read_csv(input_file, dtype=str)
    original_count = len(data)
    logger.info(f"Original dataset: {original_count} sequences")
    
    # Convert sequences to uppercase (like in R code)
    data['sequence'] = data['sequence'].str.upper()
    logger.info("Converted sequences to uppercase")
    
    # Filter by sequence length (290-320 bp) - matching R code
    logger.info("Filtering sequences by length (290-320 bp)")
    len_before = len(data)
    data = data[data['sequence'].str.len().between(290, 320)]
    logger.info(f"After length filtering: {len(data)} sequences ({len(data)/len_before:.2%} kept)")
    
    # Remove sequences with ambiguous nucleotides - matching R code
    logger.info("Removing sequences with ambiguous nucleotides")
    len_before = len(data)
    ambiguous_pattern = 'N|Y|R|W|K|S|M|D'
    data = data[~data['sequence'].str.contains(ambiguous_pattern)]
    logger.info(f"After removing ambiguous nucleotides: {len(data)} sequences ({len(data)/len_before:.2%} kept)")
    
    # Filter out rows with missing order-level classification - matching R code
    logger.info("Removing sequences without order-level classification")
    len_before = len(data)
    mask = ~((data['order_name'].isna() | data['order_name'] == "") & 
             (data['family_name'].isna() | data['family_name'] == "") & 
             (data['genus_name'].isna() | data['genus_name'] == "") & 
             (data['species_name'].isna() | data['species_name'] == ""))
    data = data[mask]
    logger.info(f"After filtering low-quality entries: {len(data)} sequences ({len(data)/len_before:.2%} kept)")
    
    # Level 1: Eukaryotes classification
    logger.info("Creating eukaryote classification level")
    euk_class = ['Metazoa', 'Viridiplantae', 'Fungi', 'No_euk', 'Others']
    
    # Initialize with kingdom name
    data['euk_class'] = data['kingdom_name']
    
    # Mark bacteria as No_euk
    data.loc[data['kingdom_name'].str.contains('Bacteria', na=False), 'euk_class'] = 'No_euk'
    
    # Mark everything else as Others
    data.loc[~data['euk_class'].isin(euk_class), 'euk_class'] = 'Others'
    
    # Level 2: Metazoa classification
    logger.info("Creating metazoa classification level")
    metazoa_class = ['Arthropoda', 'Chordata', 'Mollusca', 'Annelida', 
                     'Echinodermata', 'Platyhelminthes', 'Cnidaria',
                     'No_metazoa', 'Others']
    
    # Initialize with phylum name
    data['metazoa_class'] = data['phylum_name']
    
    # Mark non-Metazoa as No_metazoa
    data.loc[~data['euk_class'].str.contains('Metazoa', na=False), 'metazoa_class'] = 'No_metazoa'
    
    # Mark everything else as Others
    data.loc[~data['metazoa_class'].isin(metazoa_class), 'metazoa_class'] = 'Others'
    
    # Level 3: Arthropoda classification
    logger.info("Creating arthropoda classification level")
    arthropoda_class = ['Insecta', 'Arachnida', 'Malacostraca', 'Collembola', 
                        'Hexanauplia', 'Thecostraca', 'Branchiopoda', 'Diplopoda', 
                        'Ostracoda', 'Chilopoda', 'Pycnogonida',
                        'No_arthropoda', 'Others']
    
    # Initialize with class name
    data['arthropoda_class'] = data['class_name']
    
    # Mark non-Arthropoda as No_arthropoda
    data.loc[~data['metazoa_class'].str.contains('Arthropoda', na=False), 'arthropoda_class'] = 'No_arthropoda'
    
    # Mark everything else as Others
    data.loc[~data['arthropoda_class'].isin(arthropoda_class), 'arthropoda_class'] = 'Others'
    
    # Level 4: Insecta classification
    logger.info("Creating insecta classification level")
    # Fixed list to match R code (separate Ephemeroptera and Odonata)
    insecta_class = ['Diptera', 'Lepidoptera', 'Hymenoptera', 'Coleoptera', 'Hemiptera', 
                     'Trichoptera', 'Orthoptera', 'Ephemeroptera', 'Odonata', 'Blattodea', 
                     'Thysanoptera', 'Psocoptera', 'Plecoptera', 'Neuroptera',
                     'No_insecta', 'Others']
    
    # Initialize with order name
    data['insecta_class'] = data['order_name']
    
    # Mark non-Insecta as No_insecta
    data.loc[~data['arthropoda_class'].str.contains('Insecta', na=False), 'insecta_class'] = 'No_insecta'
    
    # Mark everything else as Others
    data.loc[~data['insecta_class'].isin(insecta_class), 'insecta_class'] = 'Others'
    
    # NOTE: We're not filtering for insects only to match the R code, which keeps all data
    
    # Print statistics about the dataset
    logger.info("\nDataset Statistics:")
    logger.info(f"Total sequences: {len(data)}")
    
    for col in ['euk_class', 'metazoa_class', 'arthropoda_class', 'insecta_class']:
        class_counts = data[col].value_counts()
        logger.info(f"\n{col} distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} ({count/len(data):.2%})")
    
    # Save the filtered data
    if output_file:
        data.to_csv(output_file, index=False)
        logger.info(f"Saved filtered data to {output_file}")
    
    return data

def main():
    # Setup logging
    logger = setup_logging()
    
    # Hardcoded input and output paths as requested
    input_file = 'data/merged/final_taxonomy_trimmed_coi.csv'
    output_file = 'data/merged/insects_trimmed_coi.csv'
    
    logger.info(f"Using hardcoded paths:")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    
    # Run the filtering WITHOUT focusing on insects to match R code
    filter_taxonomy_data(
        input_file, 
        output_file
    )
    
    logger.info("Filtering complete!")

if __name__ == '__main__':
    main()