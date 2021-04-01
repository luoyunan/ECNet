import pathlib
import numpy as np
import pandas as pd
import msgpack
import numba
from ecnet import vocab

@numba.njit
def all_sequence_pairwise_profile(args):
    '''
    Parameters
    ----------
    e: 4-d (L, L, 21, 21) np.array where e(i, j, a, b) is the epsilon values in CCMPred for
           `a` at the i-th postion and `b` at the j-th position.
           Amino acids are encoded by 0-20 using the same index in CCMPred.
    index_encoded: (N, L) array of index-encoded sequence using CCMPred index

    Returns
    -------
    encoding: (N, L, L) array of sequence encoding. Each amino acid in a sequence
    is encoded by values in the pairwise epsilon value table of CCMPred.
    '''
    e, index_encoded = args
    N, L = index_encoded.shape
    encoding = np.zeros((N, L, L))
    for k in range(N):
        for i in range(L - 1):
            a = index_encoded[k, i]
            for j in range(i + 1, L):
                b = index_encoded[k, j]
                encoding[k, i, j] = e[i, j, a, b]
        encoding[k] += encoding[k].T
    return encoding

@numba.njit
def all_sequence_singleton_profile(args):
    '''
    ei: 2-d (L, 20) np.array where e(i, a) is the epsilon values in CCMPred for
           `a` at the i-th postion.
           Amino acids are encoded by 0-20 using the same index in CCMPred.
    index_encoded: (N, L) array of index-encoded sequence using CCMPred index

    Returns
    -------
    encoding: (N, L, 1) array of sequence encoding. Each amino acid in a sequence
    is encoded by values in the singleton epsilon value table of CCMPred.    
    '''
    e, index_encoded = args
    N, L = index_encoded.shape
    encoding = np.zeros((N, L, 1))
    for k in range(N):
        for i in range(L):
            a = index_encoded[k, i]
            encoding[k, i] = e[i, a]
    return encoding

class CCMPredEncoder(object):
    def __init__(self, ccmpred_output=None, seq_len=None):
        '''
        brawfile: path to msgpack file storing the (L, L, 21, 21) table of
                  CCMPred epsilon values of a target sequence
        '''
        self.seq_len = seq_len
        self.vocab_index = vocab.CCMPRED_AMINO_ACID_INDEX
        brawfile = pathlib.Path(ccmpred_output)
        self.eij, self.ei = self.load_data(brawfile)

    def load_data(self, brawfile):
        '''
        Returns
        -------
        eij: 4-d (L, L, 21, 21) np.array where e(i, j, a, b) is the epsilon values in CCMPred for
           `a` at the i-th postion and `b` at the j-th position.
        ei: 2-d (L, 20) np.array where e(i, a) is the epsilon values in CCMPred for
           `a` at the i-th postion
           Amino acids are encoded by 0-20 using the same index in CCMPred.
        '''
        if not brawfile.exists():
            raise FileNotFoundError(brawfile)
        data = msgpack.unpack(open(brawfile, 'rb'))
        L = self.seq_len
        V = len(self.vocab_index)
        eij = np.zeros((L, L, V, V))
        for i in range(L - 1):
            for j in range(i + 1, L):
                arr = np.array(data[b'x_pair'][b'%d/%d'%(i, j)][b'x']).reshape(V, V)
                eij[i, j] = arr
                eij[j, i] = arr.T

        ei = np.array(data[b'x_single']).reshape(L, V - 1)

        return eij, ei


    def index_encoding(self, sequences, letter_to_index_dict):
        '''
        Modified from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/amino_acid.py#L110-L130

        Parameters
        ----------
        sequences: list of equal-length sequences
        letter_to_index_dict: char -> int

        Returns
        -------
        np.array with shape (#sequences, length of sequences)
        '''
        df = pd.DataFrame(iter(s) for s in sequences)
        encoding = df.replace(letter_to_index_dict)
        encoding = encoding.values.astype(np.int)
        return encoding

    def ccmpred_encoding(self, index_encoded, profile='pair'):
        if profile == 'pair':
            encoding = all_sequence_pairwise_profile((self.eij, index_encoded))
        elif profile == 'single':
            encoding = all_sequence_singleton_profile((self.ei, index_encoded))
        else:
            raise NotImplementedError
        return encoding

    def encode(self, sequences, mode='train'):
        """
        Returns
        -------
        encoding: (N, L, L + 1) array of sequence encoding. Each amino acid in a sequence
        is encoded by values in the singble and pairwise epsilon value tables of CCMPred
        """
        index_encoded = self.index_encoding(sequences, self.vocab_index)        
        single = self.ccmpred_encoding(index_encoded, profile='single')
        pair = self.ccmpred_encoding(index_encoded, profile='pair')
        self.ccmpred_encoded = np.concatenate([single, pair], axis=2)        
        return self.ccmpred_encoded

if __name__ == "__main__":
    from data import Dataset
    protein_name = 'TEM1'
    dataset_name = 'Envision_Gray2018'
    dataset = Dataset(
        train_tsv=f'../../output/mutagenesis/{dataset_name}/{protein_name}/data.tsv',
        fasta=f'../../output/mutagenesis/{dataset_name}/{protein_name}/native_sequence.fasta')
    print(dataset.native_sequence)
    print(dataset.train_df.head())
    encoder = CCMPredEncoder(protein_name=protein_name, dataset_name=dataset_name,
        ccmpred_output=f'../../output/homologous/{dataset_name}/{protein_name}/hhblits/ccmpred/{protein_name}.braw',
        seq_len=len(dataset.native_sequence))
    import time
    t_st = time.time()
    _ = encoder.encode(dataset.train_df['sequence'].values)
    print(time.time() - t_st)