import pandas as pd
import os

# Constants
TAXONOMY_RANKS = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
RANK_MAPPINGS = {
    'kingdom': ['subkingdom', 'superphylum'],
    'phylum': ['subphylum', 'superclass'],
    'class': ['subclass', 'infraclass'],
    'order': ['suborder', 'infraorder', 'parvorder'],
    'family': ['subfamily', 'supergenus'],
    'genus': ['subgenus'],
    'species': ['subspecies', 'species group', 'species subgroup', 'forma', 'varietas']
}

def load_sequence_data(file_path):
    """Load sequence data from TSV file."""
    print(f"Loading sequence data from {file_path}...")
    df = pd.read_csv(file_path, sep='\t', header=0, 
                    names=['seqID', 'taxID', 'sequence'], dtype=str)
    df['seqID'] = df['seqID'].astype(str)
    df['taxID'] = df['taxID'].astype(str)
    df['sequence'] = df['sequence'].astype(str).str.upper()
    return df

def load_taxonomy_data(names_path, nodes_path):
    """Load taxonomy data from NCBI taxonomy dump files."""
    print(f"Loading taxonomy names from {names_path}...")
    names_df = pd.read_csv(names_path, sep='\t\|\t', 
                          engine='python',
                          header=None, 
                          names=['tax_id', 'name_txt', 'unique_name', 'name_class'])
    names_df['name_class'] = names_df['name_class'].str.replace('\t\|$', '', regex=True)
    
    print(f"Loading taxonomy nodes from {nodes_path}...")
    nodes_df = pd.read_csv(nodes_path, sep='\t\|\t', 
                          engine='python',
                          header=None, 
                          names=['tax_id', 'parent_tax_id', 'rank', 'embl_code', 'division_id', 
                                 'inherited_div_flag', 'genetic_code_id', 'inherited_GC_flag',
                                 'mitochondrial_genetic_code_id', 'inherited_MGC_flag',
                                 'GenBank_hidden_flag', 'hidden_subtree_root_flag', 'comments'])
    nodes_df['comments'] = nodes_df['comments'].str.replace('\t\|$', '', regex=True)
    
    return names_df, nodes_df

def create_taxonomy_mappings(names_df, nodes_df):
    """Create dictionary mappings for taxonomy data."""
    sci_names = names_df[names_df['name_class'] == 'scientific name']
    sci_names_dict = dict(zip(sci_names['tax_id'].astype(str), sci_names['name_txt']))
    
    parent_dict = dict(zip(nodes_df['tax_id'].astype(str), nodes_df['parent_tax_id'].astype(str)))
    rank_dict = dict(zip(nodes_df['tax_id'].astype(str), nodes_df['rank']))
    
    return sci_names_dict, parent_dict, rank_dict

def get_lineage_with_approximation(taxid, sci_names_dict, parent_dict, rank_dict):
    """Get taxonomic lineage for a given taxid with rank approximation."""
    lineage = {rank: None for rank in TAXONOMY_RANKS}
    extended_lineage = {}
    
    current_id = str(taxid)
    visited = set()
    
    while current_id and current_id not in visited:
        visited.add(current_id)
        rank = rank_dict.get(current_id)
        name = sci_names_dict.get(current_id)
        
        if name:
            if rank:
                extended_lineage[rank] = name
            
            if rank in lineage:
                lineage[rank] = name
        
        # Move to parent
        parent_id = parent_dict.get(current_id)
        if not parent_id or parent_id == current_id:
            break
        current_id = parent_id
    
    # Try to approximate missing ranks
    for main_rank, proximal_ranks in RANK_MAPPINGS.items():
        if lineage[main_rank] is None:  # This standard rank is missing
            # Try to fill it from any available proximal rank
            for proximal_rank in proximal_ranks:
                if proximal_rank in extended_lineage:
                    lineage[main_rank] = f"{extended_lineage[proximal_rank]} (from {proximal_rank})"
                    break
    
    return lineage

def build_taxonomic_dataframe(seq_df, sci_names_dict, parent_dict, rank_dict):
    """Build the final dataframe with full taxonomic information."""
    merged_df = seq_df.copy()
    
    merged_df['scientific_name'] = merged_df['taxID'].map(sci_names_dict)
    merged_df['rank'] = merged_df['taxID'].map(rank_dict)
    
    for rank in TAXONOMY_RANKS:
        merged_df[f'{rank}_name'] = None
    
    print("\nProcessing taxonomic lineage for each sequence...")
    lineage_cache = {}
    for taxid in merged_df['taxID'].unique():
        lineage_cache[taxid] = get_lineage_with_approximation(taxid, sci_names_dict, parent_dict, rank_dict)
    
    for idx, row in merged_df.iterrows():
        taxid = row['taxID']
        lineage = lineage_cache.get(taxid, {})
        
        for rank, name in lineage.items():
            merged_df.at[idx, f'{rank}_name'] = name
    
    final_columns = ['seqID', 'taxID', 'scientific_name', 'sequence'] + [f'{rank}_name' for rank in TAXONOMY_RANKS]
    return merged_df[final_columns]

def analyze_approximations(df):
    """Analyze how many values were approximated vs. standard vs. missing."""
    print("\nApproximation Analysis:")
    for rank in TAXONOMY_RANKS:
        # Count standard values
        standard_count = df[f'{rank}_name'].notna() & ~df[f'{rank}_name'].str.contains('from', na=False)
        # Count approximated values
        approx_count = df[f'{rank}_name'].str.contains('from', na=False)
        # Count remaining missing values
        missing_count = df[f'{rank}_name'].isna()
        
        print(f"{rank}: {standard_count.sum()} standard, {approx_count.sum()} approximated, {missing_count.sum()} missing")
    
    # Calculate overall coverage
    total_cells = len(df) * len(TAXONOMY_RANKS)
    filled_cells = df[[f'{rank}_name' for rank in TAXONOMY_RANKS]].notna().sum().sum()
    print(f"\nOverall taxonomic coverage: {filled_cells/total_cells:.2%}")

def main():
    """Main function to process the taxonomy data."""
    # File paths
    data_dir = 'data/raw'
    output_dir = 'data/merged'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    seq_df = load_sequence_data(f'{data_dir}/trimmed.tsv')
    seq_df.to_csv(f'{data_dir}/trimmed.csv', index=False)
    
    names_df, nodes_df = load_taxonomy_data(f'{data_dir}/names.dmp', f'{data_dir}/nodes.dmp')
    names_df.to_csv(f'{data_dir}/names.csv', index=False)
    nodes_df.to_csv(f'{data_dir}/nodes.csv', index=False)
    
    # Create taxonomy mappings
    sci_names_dict, parent_dict, rank_dict = create_taxonomy_mappings(names_df, nodes_df)
    
    # Process taxonomic information
    final_df = build_taxonomic_dataframe(seq_df, sci_names_dict, parent_dict, rank_dict)
    
    # Save results
    output_path = f'{output_dir}/filled_blanks.csv'
    print(f"\nSaving results to {output_path}")
    final_df.to_csv(output_path, index=False)
    
    # Analyze results
    analyze_approximations(final_df)

if __name__ == "__main__":
    main()