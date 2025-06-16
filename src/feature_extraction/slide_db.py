# this script reads the database in csv format and for those sequences longer
# than 299 characters, create all the possible combinations by taking the inner sequence 
# by sliding one position at a time. For each new combination, keep the id label withan extra _n being n the number of the sequence
import pandas as pd
from dataset.parquet_builder import ParquetBuilder

def slide_db(df: pd.DataFrame, window_size: int = 300) -> pd.DataFrame:
    """
    Slide through the sequences in the DataFrame and create new rows for each sliding window.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'sequence' columns.
        window_size (int): Size of the sliding window.

    Returns:
        pd.DataFrame: New DataFrame with sliding windows.
    """
    new_rows = []
    
    for index, row in df.iterrows():
        sequence = row['sequence']
        seq_id = row['seqID']
        
        if len(sequence) > window_size:
            for i in range(len(sequence) - window_size + 1):
                row_to_append = row.to_dict()
                row_to_append['sequence'] = sequence[i:i + window_size]
                row_to_append['seqID'] = f"{seq_id}_{i}"
                new_rows.append(row_to_append)
        elif len(sequence) == window_size:
            new_rows.append(row.to_dict())
    
    return pd.DataFrame(new_rows)

if __name__ == "__main__":
    
    df =  pd.read_csv('data/database.csv') 
    result_df = slide_db(df, window_size=300)

    # for those sequences that are identical and with the same species name, keep only one
    result_df = result_df.drop_duplicates(subset=['sequence', 'scientific_name'], keep='first')

    result_df.to_csv('data/slided_data/database.csv')

    p = ParquetBuilder(csv_path='data/slided_data/database.csv',)
    p.create_parquets(parallelize=False) # With parallelize=False, it takes less than 20 minutes.
    p.show_info_parquets()