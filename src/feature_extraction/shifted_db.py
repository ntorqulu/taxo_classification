# this script reads the database in csv format and for those sequences longer
# than 299 characters, create all the possible combinations by taking the inner sequence 
# by sliding one position at a time. For each new combination, keep the id label withan extra _n being n the number of the sequence
import pandas as pd
import numpy as np
import random
from dataset.parquet_builder import ParquetBuilder

def shifted_db(df: pd.DataFrame, window_size: int = 300, resample_to: str = 'order_name') -> pd.DataFrame:
    """
    Slide through the sequences in the DataFrame and create new rows for each sliding window.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'sequence' columns.
        window_size (int): Size of the sliding window.

    Returns:
        pd.DataFrame: New DataFrame with sliding windows.
    """
    if resample_to not in ['order_name', 'class_name', 'phylum_name', 'kingdom_name']:
        raise ValueError("resample_to must be one of 'order_name', 'class_name', 'phylum_name', or 'kingdom_name'.")

    count_occurrences = df[resample_to].value_counts()
    # drop the no_insecta from count_occurrences
    if resample_to == 'order_name':
        no_label =  'No_insecta'
    elif resample_to == "class_name":
        no_label = 'No_arthropoda'
    elif resample_to == "phylum_name":
        no_label = 'No_metazoa'
    elif resample_to == "kingdom_name":
        no_label = 'No_eukaryota'
    count_occurrences = count_occurrences[count_occurrences.index != no_label]
    max_occurrences = count_occurrences.max()
    count_ratios = np.round(max_occurrences / count_occurrences )
    new_rows = []
    
    for index, row in df.iterrows():
        sequence = row['sequence']
        seq_id = row['seqID']
        
        if len(sequence) > window_size:
            number_of_windows = len(sequence) - window_size + 1
            range_of_windows = range(number_of_windows)
            if row[resample_to] == no_label:
                number_to_sample = 1
            else:
                number_to_sample  = int(count_ratios[row[resample_to]])
            if number_to_sample > number_of_windows:
                number_to_sample = number_of_windows
            selected_windows = random.sample(range_of_windows,number_to_sample)
            for i in selected_windows:
                row_to_append = row.to_dict()
                row_to_append['sequence'] = sequence[i:i + window_size]
                row_to_append['seqID'] = f"{seq_id}_{i}"
                new_rows.append(row_to_append)
        elif len(sequence) == window_size:
            new_rows.append(row.to_dict())
    
    return pd.DataFrame(new_rows)

if __name__ == "__main__":
    
    df =  pd.read_csv('data/database.csv') 
    result_df = shifted_db(df, window_size=300, resample_to='order_name')

    # for those sequences that are identical and with the same species name, keep only one
    result_df = result_df.drop_duplicates(subset=['sequence', 'scientific_name'], keep='first')

    result_df.to_csv('data/shifted_data/database.csv')

    p = ParquetBuilder(csv_path='data/shifted_data/database.csv',)
    p.create_parquets(parallelize=False) # With parallelize=False, it takes less than 20 minutes.
    p.show_info_parquets()