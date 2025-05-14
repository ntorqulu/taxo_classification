import os
import pandas as pd
import subprocess
from typing import Dict, List, Optional, Tuple, Union
import logging

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

class SequenceFormatter:
    """Class for handling the initial processing and formatting of DNA sequence data."""
    
    def __init__(self, 
                 raw_data_dir: str = 'data/raw',
                 interim_data_dir: str = 'data/interim',
                 processed_data_dir: str = 'data/processed',
                 logger=None):
        """
        Initialize the SequenceFormatter with directory paths.
        
        Parameters:
        -----------
        raw_data_dir : str
            Directory containing raw data files
        interim_data_dir : str
            Directory for intermediate processing files
        processed_data_dir : str
            Directory for final processed output files
        logger : logging.Logger, optional
            Logger for output messages, creates new one if None
        """
        self.raw_data_dir = raw_data_dir
        self.interim_data_dir = interim_data_dir
        self.processed_data_dir = processed_data_dir
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.interim_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logger or logging.getLogger(__name__)
    
    def select_coi_region(self, 
                         input_tsv: str,
                         perl_script_path: str,
                         output_dir: Optional[str] = None,
                         forward_primer: str = 'GGWACWRGWTGRACWNTNTAYCCYCC',
                         reverse_primer: str = 'TANACYTCNGGRTGNCCRAARAAYCA',
                         min_length: int = 299,
                         max_length: int = 320,
                         e_pcr: int = 1) -> str:
        """
        Run the Perl script to select COI sequences from the input TSV.
        
        Parameters:
        -----------
        input_tsv : str
            Path to input TSV file with sequence data
        perl_script_path : str
            Path to the Perl script for sequence selection
        output_dir : str, optional
            Directory for output files, defaults to raw_data_dir
        forward_primer : str
            Forward primer sequence for COI region
        reverse_primer : str
            Reverse primer sequence for COI region
        min_length : int
            Minimum amplicon length
        max_length : int
            Maximum amplicon length
        e_pcr : int
            E-PCR parameter
            
        Returns:
        --------
        str
            Path to the output TSV file with selected sequences
        """
        if output_dir is None:
            output_dir = self.raw_data_dir
            
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the command
        cmd = [
            "perl", perl_script_path,
            "-tsv", input_tsv,
            "-outdir", output_dir,
            "-fw", forward_primer,
            "-rv", reverse_primer,
            "-e_pcr", str(e_pcr),
            "-min_amplicon_length", str(min_length),
            "-max_amplicon_length", str(max_length)
        ]
        
        # Run the command
        self.logger.info(f"Running COI selection: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("COI selection completed successfully")
            self.logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running COI selection: {e}")
            self.logger.error(e.stderr)
            raise
        
        # The output file should be trimmed.tsv in the output directory
        output_file = os.path.join(output_dir, "trimmed.tsv")
        if not os.path.exists(output_file):
            self.logger.error(f"Expected output file {output_file} not found")
            raise FileNotFoundError(f"Expected output file {output_file} not found")
            
        return output_file
    
    def load_sequence_data(self, file_path: str) -> pd.DataFrame:
        """
        Load sequence data from TSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the TSV file with sequence data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sequence data
        """
        self.logger.info(f"Loading sequence data from {file_path}...")
        
        try:
            df = pd.read_csv(file_path, sep='\t', header=0, 
                            names=['seqID', 'taxID', 'sequence'], dtype=str)
            df['seqID'] = df['seqID'].astype(str)
            df['taxID'] = df['taxID'].astype(str)
            df['sequence'] = df['sequence'].astype(str).str.upper()
            
            self.logger.info(f"Loaded {len(df)} sequences")
            return df
        except Exception as e:
            self.logger.error(f"Error loading sequence data: {e}")
            raise
    
    def load_taxonomy_data(self, names_path: str, nodes_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load taxonomy data from NCBI taxonomy dump files.
        
        Parameters:
        -----------
        names_path : str
            Path to the names.dmp file
        nodes_path : str
            Path to the nodes.dmp file
            
        Returns:
        --------
        Tuple[pandas.DataFrame, pandas.DataFrame]
            Tuple of DataFrames (names_df, nodes_df)
        """
        self.logger.info(f"Loading taxonomy names from {names_path}...")
        try:
            names_df = pd.read_csv(names_path, sep='\t\|\t', 
                                  engine='python',
                                  header=None, 
                                  names=['tax_id', 'name_txt', 'unique_name', 'name_class'])
            names_df['name_class'] = names_df['name_class'].str.replace('\t\|$', '', regex=True)
            self.logger.info(f"Loaded {len(names_df)} taxonomy names")
        except Exception as e:
            self.logger.error(f"Error loading taxonomy names: {e}")
            raise
        
        self.logger.info(f"Loading taxonomy nodes from {nodes_path}...")
        try:
            nodes_df = pd.read_csv(nodes_path, sep='\t\|\t', 
                                  engine='python',
                                  header=None, 
                                  names=['tax_id', 'parent_tax_id', 'rank', 'embl_code', 'division_id', 
                                         'inherited_div_flag', 'genetic_code_id', 'inherited_GC_flag',
                                         'mitochondrial_genetic_code_id', 'inherited_MGC_flag',
                                         'GenBank_hidden_flag', 'hidden_subtree_root_flag', 'comments'])
            nodes_df['comments'] = nodes_df['comments'].str.replace('\t\|$', '', regex=True)
            self.logger.info(f"Loaded {len(nodes_df)} taxonomy nodes")
        except Exception as e:
            self.logger.error(f"Error loading taxonomy nodes: {e}")
            raise
        
        return names_df, nodes_df
    
    def create_taxonomy_mappings(self, names_df: pd.DataFrame, nodes_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
        """
        Create dictionary mappings for taxonomy data.
        
        Parameters:
        -----------
        names_df : pandas.DataFrame
            DataFrame with taxonomy names
        nodes_df : pandas.DataFrame
            DataFrame with taxonomy nodes
            
        Returns:
        --------
        Tuple[Dict, Dict, Dict]
            Tuple of dictionaries (sci_names_dict, parent_dict, rank_dict)
        """
        self.logger.info("Creating taxonomy mappings...")
        
        # Extract scientific names
        sci_names = names_df[names_df['name_class'] == 'scientific name']
        sci_names_dict = dict(zip(sci_names['tax_id'].astype(str), sci_names['name_txt']))
        
        # Create parent and rank dictionaries
        parent_dict = dict(zip(nodes_df['tax_id'].astype(str), nodes_df['parent_tax_id'].astype(str)))
        rank_dict = dict(zip(nodes_df['tax_id'].astype(str), nodes_df['rank']))
        
        self.logger.info(f"Created mappings for {len(sci_names_dict)} taxa")
        return sci_names_dict, parent_dict, rank_dict
    
    def get_lineage_with_approximation(self, 
                                      taxid: str, 
                                      sci_names_dict: Dict[str, str], 
                                      parent_dict: Dict[str, str], 
                                      rank_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Get taxonomic lineage for a given taxid with rank approximation.
        
        Parameters:
        -----------
        taxid : str
            Taxonomy ID to look up
        sci_names_dict : Dict[str, str]
            Mapping of tax_id to scientific name
        parent_dict : Dict[str, str]
            Mapping of tax_id to parent tax_id
        rank_dict : Dict[str, str]
            Mapping of tax_id to rank
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping ranks to names in the lineage
        """
        lineage = {rank: None for rank in TAXONOMY_RANKS}
        extended_lineage = {}
        
        current_id = str(taxid)
        visited = set()
        
        # Traverse up the taxonomy tree
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
    
    def build_taxonomic_dataframe(self, 
                                 seq_df: pd.DataFrame, 
                                 sci_names_dict: Dict[str, str], 
                                 parent_dict: Dict[str, str], 
                                 rank_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Build the final dataframe with full taxonomic information.
        
        Parameters:
        -----------
        seq_df : pandas.DataFrame
            DataFrame with sequence data
        sci_names_dict : Dict[str, str]
            Mapping of tax_id to scientific name
        parent_dict : Dict[str, str]
            Mapping of tax_id to parent tax_id
        rank_dict : Dict[str, str]
            Mapping of tax_id to rank
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sequences and complete taxonomic lineage
        """
        self.logger.info("Building taxonomic dataframe...")
        merged_df = seq_df.copy()
        
        # Add scientific name and rank
        merged_df['scientific_name'] = merged_df['taxID'].map(sci_names_dict)
        merged_df['rank'] = merged_df['taxID'].map(rank_dict)
        
        # Initialize columns for taxonomy ranks
        for rank in TAXONOMY_RANKS:
            merged_df[f'{rank}_name'] = None
        
        # Process taxonomic lineage for each unique taxID
        self.logger.info("Processing taxonomic lineage for each sequence...")
        lineage_cache = {}
        
        # First build a cache of lineages
        for taxid in merged_df['taxID'].unique():
            lineage_cache[taxid] = self.get_lineage_with_approximation(
                taxid, sci_names_dict, parent_dict, rank_dict)
        
        # Then fill in the dataframe
        for idx, row in merged_df.iterrows():
            taxid = row['taxID']
            lineage = lineage_cache.get(taxid, {})
            
            for rank, name in lineage.items():
                merged_df.at[idx, f'{rank}_name'] = name
        
        # Select the final columns in desired order
        final_columns = ['seqID', 'taxID', 'scientific_name', 'sequence'] + [f'{rank}_name' for rank in TAXONOMY_RANKS]
        final_df = merged_df[final_columns]
        
        self.logger.info(f"Completed taxonomic dataframe with {len(final_df)} sequences")
        return final_df
    
    def analyze_approximations(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Analyze how many values were approximated vs. standard vs. missing.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with taxonomic information
            
        Returns:
        --------
        Dict[str, Dict[str, int]]
            Dictionary with counts of standard, approximated, and missing values for each rank
        """
        self.logger.info("Analyzing taxonomic approximations...")
        results = {}
        
        for rank in TAXONOMY_RANKS:
            # Count standard values
            standard_count = df[f'{rank}_name'].notna() & ~df[f'{rank}_name'].str.contains('from', na=False)
            # Count approximated values
            approx_count = df[f'{rank}_name'].str.contains('from', na=False)
            # Count remaining missing values
            missing_count = df[f'{rank}_name'].isna()
            
            results[rank] = {
                'standard': int(standard_count.sum()),
                'approximated': int(approx_count.sum()),
                'missing': int(missing_count.sum())
            }
            
            self.logger.info(f"{rank}: {standard_count.sum()} standard, {approx_count.sum()} approximated, {missing_count.sum()} missing")
        
        # Calculate overall coverage
        total_cells = len(df) * len(TAXONOMY_RANKS)
        filled_cells = df[[f'{rank}_name' for rank in TAXONOMY_RANKS]].notna().sum().sum()
        coverage = filled_cells/total_cells
        
        self.logger.info(f"Overall taxonomic coverage: {coverage:.2%}")
        results['overall_coverage'] = coverage
        
        return results
    
    def format_data(self, 
                   run_coi_selection: bool = False,
                   perl_script_path: Optional[str] = None,
                   input_tsv: Optional[str] = None,
                   names_dmp: Optional[str] = None,
                   nodes_dmp: Optional[str] = None,
                   output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Complete pipeline to format and structure DNA sequence data.
        
        Parameters:
        -----------
        run_coi_selection : bool
            Whether to run the Perl script for COI region selection
        perl_script_path : str, optional
            Path to the Perl script for COI selection
        input_tsv : str, optional
            Path to input TSV file with sequence data, if None uses default path
        names_dmp : str, optional
            Path to names.dmp file, if None uses default path
        nodes_dmp : str, optional
            Path to nodes.dmp file, if None uses default path
        output_file : str, optional
            Path for saving the formatted data, if None uses default path
            
        Returns:
        --------
        pandas.DataFrame
            Formatted DataFrame with sequences and taxonomic information
        """
        # Set default paths if not provided
        if input_tsv is None:
            input_tsv = os.path.join(self.raw_data_dir, "NJORDR_sequences.tsv")
        
        if names_dmp is None:
            names_dmp = os.path.join(self.raw_data_dir, "names.dmp")
        
        if nodes_dmp is None:
            nodes_dmp = os.path.join(self.raw_data_dir, "nodes.dmp")
        
        if output_file is None:
            output_file = os.path.join(self.processed_data_dir, "formatted_sequences.csv")
        
        # Step 1: Run COI selection if requested
        if run_coi_selection:
            if perl_script_path is None:
                raise ValueError("perl_script_path must be provided when run_coi_selection is True")
            
            self.select_coi_region(
                input_tsv=input_tsv,
                perl_script_path=perl_script_path,
                output_dir=self.raw_data_dir
            )
            
            # Update input_tsv to point to the trimmed file
            input_tsv = os.path.join(self.raw_data_dir, "trimmed.tsv")
        
        # Step 2: Load sequence data
        seq_df = self.load_sequence_data(input_tsv)
        
        # Step 3: Load taxonomy data
        names_df, nodes_df = self.load_taxonomy_data(names_dmp, nodes_dmp)
        
        # Step 4: Create taxonomy mappings
        sci_names_dict, parent_dict, rank_dict = self.create_taxonomy_mappings(names_df, nodes_df)
        
        # Step 5: Build taxonomic dataframe
        final_df = self.build_taxonomic_dataframe(seq_df, sci_names_dict, parent_dict, rank_dict)
        
        # Step 6: Analyze and report approximations
        approximation_stats = self.analyze_approximations(final_df)
        
        # Step 7: Save the formatted data
        self.logger.info(f"Saving formatted data to {output_file}")
        final_df.to_csv(output_file, index=False)
        
        return final_df

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    formatter = SequenceFormatter()
    
    # Example paths (update these to your actual file paths)
    perl_script_path = "src/preprocessing/select_region.pl"
    input_tsv = "data/raw/NJORDR_sequences.tsv"
    names_dmp = "data/raw/names.dmp"
    nodes_dmp = "data/raw/nodes.dmp"
    
    # Run the formatter
    formatted_df = formatter.format_data(
        run_coi_selection=True,
        perl_script_path=perl_script_path,
        input_tsv=input_tsv,
        names_dmp=names_dmp,
        nodes_dmp=nodes_dmp
    )
    print(formatted_df.head())