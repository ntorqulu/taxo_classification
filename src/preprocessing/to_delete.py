# read NJORDR_sequences.tsv
# just get the first 20 rows
import pandas as pd
import os

# Path to the original sequence file
input_file = "data/raw/NJORDR_sequences.tsv"
# Path to save the smaller test file
output_file = "data/raw/test_sequences.tsv"

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Read the first n rows of the TSV file
n_rows = 20  # You can adjust this number as needed
try:
    # Try to read with tab separator
    df = pd.read_csv(input_file, sep='\t', nrows=n_rows)
    print(f"Successfully read {len(df)} rows with tab separator")
except Exception as e:
    # If that fails, try to detect the separator
    print(f"Error reading with tab separator: {e}")
    print("Trying to detect separator...")
    df = pd.read_csv(input_file, nrows=n_rows)
    print(f"Successfully read {len(df)} rows with auto-detected separator")

# Print column names and a sample row to verify
print(f"Columns: {df.columns.tolist()}")
print("\nSample row:")
print(df.iloc[0])

# Save the subset to a new TSV file
df.to_csv(output_file, sep='\t', index=False)
print(f"\nSaved {len(df)} rows to {output_file}")

# Display the command to run the Perl script with this test file
perl_script = "src/preprocessing/select_region.pl"
cmd = f"perl {perl_script} -tsv {output_file} -outdir data/raw -fw GGWACWRGWTGRACWNTNTAYCCYCC -rv TANACYTCNGGRTGNCCRAARAAYCA -e_pcr 1 -min_amplicon_length 299 -max_amplicon_length 320"

print("\nTest the Perl script with this command:")
print(cmd)