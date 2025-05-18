import pandas as pd
import numpy as np
import torch

class SequenceCoder:
    LETTERS = ('A', 'T', 'C', 'G')

    def load_sequences(self, file_path):
        """
        Load sequences from a CSV file.
        The CSV file should have a column named 'sequence'.
        """
        self.entries = pd.read_csv(file_path)
        self.sequences = list(self.entries.sequence)
        # Display basic statistics about the dataset
        print(f"Number of entries: {len(self.sequences)}")

    # the code for the kmer is based on a modification from https://github.com/MindAI/kmer/

    def kmerize_one_seq(self, sequence: str, k: int, write_number_of_occurrences: bool = True) -> np.ndarray:
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.

        Args:
        seq: 
            a string, a DNA sequence
        write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(sequence) - k + 1
        n = 4**k # number of possible k-mers

        self.multiplyBy = 4 ** np.arange(k-1, -1, -1) # the multiplying number for each digit position in the k-number system

        kmer_feature = np.zeros(n)

        for i in range(number_of_kmers):
            this_kmer = sequence[i:(i+k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature
    
    def kmer_numbering_for_one_kmer(self, kmer):
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
        """
        digits = []
        for letter in kmer:
            digits.append(SequenceCoder.LETTERS.index(letter))

        # digits = np.array(digits)
        digits = torch.tensor(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering

    def coding_kmer(self,sequences = None, k = 1, write_number_of_occurrences=True):
        """
        extract the features from the sequences, code the sequences.
        If sequences is None, it will use the sequences loaded from the CSV file.
        """
        if sequences is None:
            if self.sequences is None:
                raise ValueError("No sequences loaded. Please load sequences first.")
        else:
            self.sequences = sequences
        # Example feature extraction: length of each sequence
        kmer_features = []
        for seq in self.sequences:
            this_kmer_feature = self.kmerize_one_seq(seq.upper(), k, write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.append(this_kmer_feature)

        kmer_features = np.array(kmer_features)
        kmer_features = torch.tensor(kmer_features, dtype=torch.float32)

        return kmer_features
    
    def oneseq_to_matrix(self, sequence):
        # Define the mapping for each nucleotide to a channel
        mapping = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
        
        # Initialize a zero tensor of shape (4, len(sequence))
        tensor = np.zeros((4, len(sequence)))
        
        # One-hot encode the sequence
        for i, nucleotide in enumerate(sequence):
            if nucleotide in mapping:
                tensor[mapping[nucleotide], i] = 1.0
            else:
                raise ValueError(f"Invalid nucleotide '{nucleotide}' in sequence.")
        
        return tensor

    
    def coding_one_hot_4rowMatrix(self, sequences = None):
        if sequences is None:
            if self.sequences is None:
                raise ValueError("No sequences loaded. Please load sequences first.")
        else:
            self.sequences = sequences
        
        # Example feature extraction: length of each sequence
        matrix_features = []
        for seq in self.sequences:
            this_matrix_feature = self.oneseq_to_matrix(seq.upper())
            matrix_features.append(this_matrix_feature)

        return matrix_features
    
    def dna_to_bitcoding(self,sequence,bits=4):
        # Mapping of nucleotides to 4-bit binary values
        if bits == 4:
            mapping = {'A': '1000', 'C': '0100', 'T': '0010', 'G': '0001'}
        elif bits == 3:
            mapping = {'A': '000', 'C': '001', 'T': '010', 'G': '100'}
        elif bits == 2:
            mapping = {'A': '00', 'C': '01', 'T': '10', 'G': '11'}
        elif bits == 1:
            mapping = {'A': '1', 'C': '2', 'T': '3', 'G': '4'}
        
        # Encode the sequence
        number_string = [mapping[nucleotide] for nucleotide in sequence]
        # Join the binary strings into a single string
        number_string = ''.join(number_string)
        
        # Split each 4-bit binary value into individual digits
        digit_list = [int(char) for char in number_string]
        digit_tensor = torch.tensor(digit_list, dtype=torch.float32)
        
        return digit_tensor

    def coding_one_hot_bit(self, sequences = None, bits = 4):
        if sequences is None:
            if self.sequences is None:
                raise ValueError("No sequences loaded. Please load sequences first.")
        else:
            self.sequences = sequences
        
        # Example feature extraction: length of each sequence
        bits_features = []
        for seq in self.sequences:
            this_bits_feature = self.dna_to_bitcoding(seq.upper(),bits)
            bits_features.append(this_bits_feature)

        return bits_features


def kmer_encoder(sequence: str, k: int) -> np.ndarray:
    kmer = SequenceCoder().kmerize_one_seq(sequence, k)
    return kmer


    
if __name__ == "__main__":
    # if this script is run directly, it runs an example of how it works

    # the output formats can be the following:
    # i.e. for seq ACGTCG:
    # kmer
        # 1mer:
            # A: 1
            # C: 2
            # G: 2
            # T: 1
        # 2mer:
            # AA: 0
            # AC: 1
            # AG: 0
            # AT: 0
            # CA: 0
            # CC: 0
            # CG: 2
            # CT: 0
            # GA: 0
            # GC: 0
            # GG: 0
            # GT: 1
            # TA: 0
            # TC: 1
            # TG: 0
            # TT: 0
        # 3mer:
            # ACG: 1
            # ...
    # one-hot-encoding:
        # matrix of 4 x N 
            # 1 0 0 0 0 0 # A
            # 0 1 0 0 1 0 # C
            # 0 0 0 1 0 0 # T
            # 0 0 1 0 0 1 # G
        # in just one line 4 digits per letter
            # 1000 0100 0010 0001 0100 0010
        # 2 digits
            # 00 01 10 11 01 00

    print("Example usage of SequenceCoder")

    sequences = SequenceCoder()
    sequences.load_sequences("data/small_dataset.csv")
    print('output of the kmer coding when loading data from a file')
    print('For k = 1:')
    print(sequences.coding_kmer(k=1))
    print('For k = 2:')
    print(sequences.coding_kmer(k=2))
    print('For k = 3:')
    print(sequences.coding_kmer(k=3))
    print('For k = 4:')
    print(sequences.coding_kmer(k=4))

    print('One-hot-encoding:')
    print('as a matrix:')
    print(sequences.coding_one_hot_4rowMatrix())
    print('as a 1D array:')
    print('4 digits per letter:')
    print(sequences.coding_one_hot_bit(bits=4))
    print('3 digits per letter:')
    print(sequences.coding_one_hot_bit(bits=3))
    print('2 digits per letter:')
    print(sequences.coding_one_hot_bit(bits=2))
    print('1 digit per letter:')
    print(sequences.coding_one_hot_bit(bits=1))

    # externally loaded:
    print()
    file_path = "data/small_dataset.csv"
    entries = pd.read_csv(file_path)
    sequences = list(entries.sequence)
    coder = SequenceCoder()
    print('output of the kmer coding when loading data from a file')
    print('For k = 1:')
    print(coder.coding_kmer(sequences = sequences, k=1))
    print('For k = 2:')
    print(coder.coding_kmer(sequences = sequences, k=2))
    print('For k = 3:')
    print(coder.coding_kmer(sequences = sequences, k=3))
    print('For k = 4:')
    print(coder.coding_kmer(sequences = sequences, k=4))
    print('One-hot-encoding:')
    print('as a matrix:')
    print(coder.coding_one_hot_4rowMatrix(sequences = sequences))
    print('as a 1D array:')
    print('4 digits per letter:')
    print(coder.coding_one_hot_bit(sequences = sequences, bits=4))
    print('3 digits per letter:')
    print(coder.coding_one_hot_bit(sequences = sequences, bits=3))
    print('2 digits per letter:')
    print(coder.coding_one_hot_bit(sequences = sequences, bits=2))
    print('1 digit per letter:')
    print(coder.coding_one_hot_bit(sequences = sequences, bits=1))