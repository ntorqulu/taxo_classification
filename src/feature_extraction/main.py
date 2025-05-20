import pandas as pd
import numpy as np
import torch
import multiprocessing as mp
from functools import partial
import time

class SequenceCoder:
    LETTERS = ['A', 'T', 'C', 'G']

    def __init__(self):
        """
        Initialize the sequence coder with basic parameters.
        """
        self.sequences = None
        self.init_lookup_tables()
    
    def init_lookup_tables(self):
        """Initialize lookup tables for faster encoding"""
        # For bit encoding
        self.bit_mapping = {
            4: {base: np.array([int(bit) for bit in bits], dtype=np.float32) 
                for base, bits in {'A': '1000', 'C': '0100', 'T': '0010', 'G': '0001'}.items()},
            3: {base: np.array([int(bit) for bit in bits], dtype=np.float32) 
                for base, bits in {'A': '100', 'C': '010', 'T': '001', 'G': '000'}.items()},
            2: {base: np.array([int(bit) for bit in bits], dtype=np.float32) 
                for base, bits in {'A': '00', 'C': '01', 'T': '10', 'G': '11'}.items()},
            1: {base: np.array([int(bit) for bit in bit], dtype=np.float32) 
                for base, bit in {'A': '1', 'C': '2', 'T': '3', 'G': '4'}.items()}
        }
    
    def load_sequences(self, file_path, verbose=True):
        """
        Load sequences from a CSV file with progress reporting for large files.
        The CSV file should have a column named 'sequence'.
        """
        start_time = time.time()
        if verbose:
            print(f"Loading sequences from {file_path}...")
        
        # Use chunk loading for very large files
        if file_path.endswith('.csv'):
            chunks = pd.read_csv(file_path, chunksize=100000)
            sequence_chunks = []
            for i, chunk in enumerate(chunks):
                if verbose and i % 10 == 0:
                    print(f"Processed {i*100000} sequences...")
                sequence_chunks.append(chunk['sequence'])
            self.entries = pd.concat(sequence_chunks)
            self.sequences = list(self.entries)
        else:
            self.entries = pd.read_csv(file_path)
            self.sequences = list(self.entries.sequence)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"Loaded {len(self.sequences)} sequences in {elapsed:.2f} seconds")

    # ===== K-MER ENCODING OPTIMIZATIONS =====
    
    def kmerize_one_seq_optimized(self, sequence: str, k: int, write_number_of_occurrences: bool=True) -> np.ndarray:
        """
        Optimized version of k-mer counting for a single sequence.
        """
        # Handle invalid sequences
        if len(sequence) < k:
            return np.zeros(4**k)
            
        # Filter out non-standard nucleotides
        valid_seq = ''.join(c for c in sequence if c in SequenceCoder.LETTERS)
        
        # Early return if sequence is too short after filtering
        if len(valid_seq) < k:
            return np.zeros(4**k)
        
        # Precompute multiplication factors
        multiply_by = 4 ** np.arange(k-1, -1, -1)
        
        # Extract all k-mers
        kmers = [valid_seq[i:i+k] for i in range(len(valid_seq)-k+1)]
        
        # Map to indices
        indices = []
        for kmer in kmers:
            if len(kmer) == k:  # Make sure it's a complete k-mer
                try:
                    # Convert bases to indices
                    digits = np.array([SequenceCoder.LETTERS.index(letter) for letter in kmer])
                    # Calculate k-mer index
                    index = int((digits * multiply_by).sum())
                    indices.append(index)
                except ValueError:
                    continue  # Skip k-mers with invalid bases
        
        # Use NumPy's bincount for fast counting
        if indices:
            counts = np.bincount(indices, minlength=4**k)
            
            # Normalize if needed
            if not write_number_of_occurrences:
                counts = counts / len(indices)
                
            return counts
        else:
            return np.zeros(4**k)
    
    def kmer_numbering_for_one_kmer(self, kmer, multiply_by):
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation).
        Now with error handling.
        """
        try:
            digits = []
            for letter in kmer:
                if letter in SequenceCoder.LETTERS:
                    digits.append(SequenceCoder.LETTERS.index(letter))
                else:
                    # Skip invalid bases
                    return -1
            
            digits = np.array(digits)
            numbering = (digits * multiply_by).sum()
            return int(numbering)  # Ensure integer for array indexing
        except:
            return -1  # Return invalid index on error
    
    def coding_kmer_optimized(self, sequences=None, k=3, write_number_of_occurrences=True,
                             batch_size=1000, n_jobs=None):
        """
        Optimized k-mer encoding with parallel processing for large datasets.
        """
        start_time = time.time()
        
        # Process sequences
        if sequences is None:
            if self.sequences is None:
                raise ValueError("No sequences loaded. Please load sequences first.")
            sequences_to_use = self.sequences
        else:
            sequences_to_use = sequences
        
        print(f"Processing {len(sequences_to_use)} sequences with k={k}...")
        
        # Determine number of parallel jobs
        if n_jobs is None:
            n_jobs = max(1, mp.cpu_count() - 1)
        
        valid_sequences = sequences_to_use
        
        if len(valid_sequences) == 0:
            return torch.zeros((len(sequences_to_use), 4**k), dtype=torch.float32)
        
        # Process in batches to avoid memory issues
        n_features = 4**k
        result = np.zeros((len(sequences_to_use), n_features), dtype=np.float32)
        
        # Process batches
        for i in range(0, len(valid_sequences), batch_size):
            batch = valid_sequences[i:min(i+batch_size, len(valid_sequences))]
            print(f"Processing batch {i//batch_size + 1}/{(len(valid_sequences)-1)//batch_size + 1}...")
            
            # Use parallel processing if batch is large enough
            if len(batch) > 100 and n_jobs > 1:
                # Split batch for parallel processing
                batch_splits = np.array_split(batch, n_jobs)
                
                # Create partial function with fixed parameters
                func = partial(self._process_kmer_batch, k=k, 
                               write_number_of_occurrences=write_number_of_occurrences)
                
                # Process in parallel
                with mp.Pool(n_jobs) as pool:
                    batch_results = pool.map(func, batch_splits)
                
                # Combine results
                batch_result = np.vstack(batch_results)
            else:
                # Process sequentially for small batches
                batch_result = self._process_kmer_batch(batch, k, write_number_of_occurrences)
            
            # Update results for valid sequences
            result[i:i+len(batch_result)] = batch_result
        
        elapsed = time.time() - start_time
        print(f"K-mer encoding completed in {elapsed:.2f} seconds")
        
        return torch.tensor(result, dtype=torch.float32)
    
    def _process_kmer_batch(self, sequences, k, write_number_of_occurrences):
        """Helper function for parallel k-mer processing"""
        batch_result = np.zeros((len(sequences), 4**k), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            try:
                batch_result[i] = self.kmerize_one_seq_optimized(
                    seq.upper(), k, write_number_of_occurrences)
            except Exception as e:
                print(f"Error processing sequence: {e}")
        
        return batch_result
    
    # Legacy method for compatibility
    def coding_kmer(self, sequences=None, k=1, write_number_of_occurrences=True):
        """Legacy k-mer encoding - redirects to optimized version"""
        return self.coding_kmer_optimized(sequences, k, write_number_of_occurrences)
    
    # ===== ONE-HOT MATRIX ENCODING OPTIMIZATIONS =====
    
    def oneseq_to_matrix_vectorized(self, sequence):
        """Vectorized version of one-hot matrix encoding"""
        # Pre-allocate matrix
        result = np.zeros((4, len(sequence)), dtype=np.float32)
        
        # Create lookup
        mapping = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
        
        # Process only valid bases
        for i, base in enumerate(sequence):
            if base in mapping:
                result[mapping[base], i] = 1.0
        
        return result
    
    def coding_one_hot_4rowMatrix_optimized(self, sequences=None, return_tensor=False,
                                          batch_size=1000, n_jobs=None):
        """
        Optimized one-hot matrix encoding with optional batching and parallelization.
        """
        start_time = time.time()
        
        # Process sequences
        if sequences is None:
            if self.sequences is None:
                raise ValueError("No sequences loaded. Please load sequences first.")
            sequences_to_use = self.sequences
        else:
            sequences_to_use = sequences
        
        print(f"Processing {len(sequences_to_use)} sequences for matrix encoding...")
        
        # Determine number of parallel jobs
        if n_jobs is None:
            n_jobs = max(1, mp.cpu_count() - 1)
        
        # Process in batches to avoid memory issues
        matrix_features = []
        
        # Process batches
        for i in range(0, len(sequences_to_use), batch_size):
            batch = sequences_to_use[i:min(i+batch_size, len(sequences_to_use))]
            print(f"Processing batch {i//batch_size + 1}/{(len(sequences_to_use)-1)//batch_size + 1}...")
            
            # Use parallel processing if batch is large enough
            if len(batch) > 100 and n_jobs > 1:
                # Split batch for parallel processing
                batch_splits = np.array_split(batch, n_jobs)
                
                # Process in parallel
                with mp.Pool(n_jobs) as pool:
                    batch_results = pool.map(self._process_matrix_batch, batch_splits)
                
                # Combine results
                for result_list in batch_results:
                    matrix_features.extend(result_list)
            else:
                # Process sequentially for small batches
                batch_result = self._process_matrix_batch(batch)
                matrix_features.extend(batch_result)
        
        elapsed = time.time() - start_time
        print(f"Matrix encoding completed in {elapsed:.2f} seconds")
        
        # Optional tensor conversion with padding
        if return_tensor and matrix_features:
            return self.pad_and_batch_matrices(matrix_features)
        
        return matrix_features
    
    def _process_matrix_batch(self, sequences):
        """Helper function for parallel matrix processing"""
        results = []
        
        for seq in sequences:
            try:
                result = self.oneseq_to_matrix_vectorized(seq.upper())
                results.append(result)
            except Exception as e:
                print(f"Error processing sequence for matrix encoding: {e}")
                # Add empty matrix to maintain sequence count
                results.append(np.zeros((4, 1), dtype=np.float32))
        
        return results
    
    def pad_and_batch_matrices(self, matrix_features):
        """
        Pad sequences to the same length and batch them into a single tensor.
        """
        # Find the maximum sequence length
        max_length = max(matrix.shape[1] for matrix in matrix_features)
        
        # Pad each matrix to the maximum length
        padded_matrices = []
        for matrix in matrix_features:
            pad_width = ((0, 0), (0, max_length - matrix.shape[1]))
            padded = np.pad(matrix, pad_width, mode='constant', constant_values=0)
            padded_matrices.append(padded)
        
        # Stack into a batch and convert to tensor
        batch = np.stack(padded_matrices, axis=0)
        return torch.tensor(batch, dtype=torch.float32)
    
    # Legacy method for compatibility
    def oneseq_to_matrix(self, sequence):
        """Legacy method - redirects to vectorized version"""
        return self.oneseq_to_matrix_vectorized(sequence)
    
    # Legacy method for compatibility
    def coding_one_hot_4rowMatrix(self, sequences=None, return_tensor=False):
        """Legacy method - redirects to optimized version"""
        return self.coding_one_hot_4rowMatrix_optimized(sequences, return_tensor)
    
    # ===== BIT ENCODING OPTIMIZATIONS =====
    
    def dna_to_bitcoding_optimized(self, sequence: str, bits: int = 4) -> np.ndarray:
        """Optimized bit encoding using pre-computed tables"""
        # Get the mapping for this bit size
        mapping = self.bit_mapping[bits]
        
        # Pre-allocate array
        result_length = len(sequence) * bits
        result = np.zeros(result_length, dtype=np.float32)
        
        # Set values
        pos = 0
        for nucleotide in sequence:
            if nucleotide in mapping:
                bit_pattern = mapping[nucleotide]
                result[pos:pos+bits] = bit_pattern
            pos += bits
            
        return result[:pos]
    
    def coding_one_hot_bit_optimized(self, sequences=None, bits=4, return_tensor=False,
                                   batch_size=1000, n_jobs=None):
        """
        Optimized bit encoding with optional batching and parallelization.
        """
        start_time = time.time()
        
        # Process sequences
        if sequences is None:
            if self.sequences is None:
                raise ValueError("No sequences loaded. Please load sequences first.")
            sequences_to_use = self.sequences
        else:
            sequences_to_use = sequences
        
        print(f"Processing {len(sequences_to_use)} sequences for bit encoding (bits={bits})...")
        
        # Determine number of parallel jobs
        if n_jobs is None:
            n_jobs = max(1, mp.cpu_count() - 1)
        
        # Process in batches to avoid memory issues
        bits_features = []
        
        # Process batches
        for i in range(0, len(sequences_to_use), batch_size):
            batch = sequences_to_use[i:min(i+batch_size, len(sequences_to_use))]
            print(f"Processing batch {i//batch_size + 1}/{(len(sequences_to_use)-1)//batch_size + 1}...")
            
            # Use parallel processing if batch is large enough
            if len(batch) > 100 and n_jobs > 1:
                # Split batch for parallel processing
                batch_splits = np.array_split(batch, n_jobs)
                
                # Create partial function with fixed parameters
                func = partial(self._process_bit_batch, bits=bits)
                
                # Process in parallel
                with mp.Pool(n_jobs) as pool:
                    batch_results = pool.map(func, batch_splits)
                
                # Combine results
                for result_list in batch_results:
                    bits_features.extend(result_list)
            else:
                # Process sequentially for small batches
                batch_result = self._process_bit_batch(batch, bits)
                bits_features.extend(batch_result)
        
        elapsed = time.time() - start_time
        print(f"Bit encoding completed in {elapsed:.2f} seconds")
        
        # Optional tensor conversion with padding
        if return_tensor and bits_features:
            return self.pad_and_batch_bit_vectors(bits_features)
        
        return bits_features
    
    def _process_bit_batch(self, sequences, bits):
        """Helper function for parallel bit encoding processing"""
        results = []
        
        for seq in sequences:
            try:
                result = self.dna_to_bitcoding_optimized(seq.upper(), bits)
                results.append(result)
            except Exception as e:
                print(f"Error processing sequence for bit encoding: {e}")
                # Add empty tensor to maintain sequence count
                results.append(torch.zeros(1, dtype=torch.float32))
        
        return results
    
    def pad_and_batch_bit_vectors(self, bit_features):
        """
        Pad bit vectors to the same length and batch them into a single tensor.
        """
        # Find the maximum length
        max_length = max(tensor.shape[0] for tensor in bit_features)
        
        # Pad each tensor to the maximum length
        padded_tensors = []
        for tensor in bit_features:
            padding = torch.zeros(max_length - tensor.shape[0], dtype=tensor.dtype)
            padded = torch.cat([tensor, padding])
            padded_tensors.append(padded)
        
        # Stack into a batch
        return torch.stack(padded_tensors, dim=0)
    
    # Legacy methods for compatibility
    def dna_to_bitcoding(self, sequence, bits=4):
        """Legacy method - redirects to optimized version"""
        return self.dna_to_bitcoding_optimized(sequence, bits)
    
    def coding_one_hot_bit(self, sequences=None, bits=4):
        """Legacy method - redirects to optimized version"""
        return self.coding_one_hot_bit_optimized(sequences, bits)
        
    # ===== ADDITIONAL UTILITY METHODS =====
    
    def process_in_batches(self, sequences=None, method='kmer', batch_size=1000, **kwargs):
        """
        Process large datasets in batches to save memory.
        Unified interface for all encoding methods.
        """
        if method == 'kmer':
            return self.coding_kmer_optimized(sequences, batch_size=batch_size, **kwargs)
        elif method == 'matrix':
            return self.coding_one_hot_4rowMatrix_optimized(sequences, batch_size=batch_size, **kwargs)
        elif method == 'bit':
            return self.coding_one_hot_bit_optimized(sequences, batch_size=batch_size, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

def kmer_encoder(sequence: str, k: int) -> np.ndarray:
    kmer = SequenceCoder().kmerize_one_seq_optimized(sequence, k)
    return kmer


# Example usage
if __name__ == "__main__":
    # Create encoder
    encoder = SequenceCoder()
    
    # Example with cleaned_sequences.csv
    try:
        # Load the data
        encoder.load_sequences("data/processed/cleaned_sequences.csv")
        
        print("\nDemonstrating different encoding methods:")
        
        # K-mer encoding (using optimized parallel version)
        start = time.time()
        kmer_features = encoder.coding_kmer_optimized(k=3, batch_size=1000, n_jobs=4)
        elapsed = time.time() - start
        print(f"K-mer encoding completed in {elapsed:.2f} seconds")
        print(f"K-mer features shape: {kmer_features.shape}")
        print(f"First 5 k-mer features: {kmer_features[:5]}")
        
        # Matrix encoding with small subset (for example purposes)
        small_subset = encoder.sequences[:100]  # Just use a few sequences
        start = time.time()
        matrix_features = encoder.coding_one_hot_4rowMatrix_optimized(
            sequences=small_subset, return_tensor=True)
        elapsed = time.time() - start
        print(f"Matrix encoding completed in {elapsed:.2f} seconds")
        print(f"Matrix shape: {matrix_features.shape}")
        print(f"First 5 matrix features: {matrix_features[:5]}")
        
        # Bit encoding with small subset
        start = time.time()
        bit_features = encoder.coding_one_hot_bit_optimized(
            sequences=small_subset, bits=2, return_tensor=True)
        elapsed = time.time() - start
        print(f"Bit encoding completed in {elapsed:.2f} seconds")
        print(f"Bit features shape: {bit_features.shape}")
        print(f"First 5 bit features: {bit_features[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
