import pandas as pd
import os

def load_taxonomy_data(filepath=None, dtype=str):
    """
    Load taxonomy dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to the CSV file. If None, uses default path.
    dtype : type or dict, optional
        Data type to use for DataFrame columns
        
    Returns:
    --------
    pandas.DataFrame
        Loaded taxonomy dataframe
    """
    if filepath is None:
        # Default path relative to project root
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'merged', 'final_taxonomy.csv'
        )
    
    # Load data
    df = pd.read_csv(filepath, dtype=dtype)
    
    # Print basic info
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Total number of sequences: {len(df)}")
    
    return df