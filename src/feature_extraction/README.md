# Feature extraction

In this folder, in the present version there are some files created by Nuria and one created by Adrià:
Adrià: I did not removed the files created by Nuria unsure if they would be better than mine. I created just one script that will be described in this README

## main.py
This script contains a class 'SequenceCoder' that contains some methods. The important ones are:

- load_sequences(file_path): to load the sequences into the class from a CSV file with the sequences in the column 'sequence'. If this option is not used, the sequences have to be specified as a parameter in the following methods

- coding_kmer(sequences = None, k = 1, write_number_of_occurrences=True): it returns a tensor with a vector for each sequence. The length of the vector is all the combinations possible at a kmer size of 'k'. vector size = 4^k. By default it counts the occurrences of each kmer but if write_number_of_occurrences=False, then the occurrences are turned into relative numbers.

- coding_one_hot_4rowMatrix(sequence = None): It returns a tensor with a matrix for each sequence of size 4 x seq_length. Each row is one of the four possible letters and in each position it gives 1 if present 0 if not.

- coding_one_hot_bit(sequences = None, bits = 4): it returns a vector for which each letter of the sequence is translated to a certain amount of digits, from 1 to 4 as follows: 

if bits == 4:\
    mapping = {'A': '1000', 'C': '0100', 'T': '0010', 'G': '0001'}\
elif bits == 3:\
    mapping = {'A': '000', 'C': '001', 'T': '010', 'G': '100'}\
elif bits == 2:\
    mapping = {'A': '00', 'C': '01', 'T': '10', 'G': '11'}\
elif bits == 1:\
    mapping = {'A': '1', 'C': '2', 'T': '3', 'G': '4'}\


# example of usage:
```
from src.feature_extraction.main import SequenceCoder

# loading the sequences from the SequenceCoder
file_path = "data/small_dataset.csv"
sequences = SequenceCoder()
sequences.load_sequences(file_path = file_path)

kmer_coded = sequences.coding_kmer(k=4)
matrix_coded = sequences.coding_one_hot_4rowMatrix()
onevect = sequences.coding_one_hot_bit(bits=4)

# With sequences as input parameter:
entries = pd.read_csv(file_path)
sequences = list(entries.sequence)
coder = SequenceCoder()

kmer_coded2 = coder.coding_kmer(sequences = sequences, k=1)
matrix_coded2 = coder.coding_one_hot_4rowMatrix(sequences = sequences)
onevect2 = coder.coding_one_hot_bit(sequences = sequences, bits=4)
```