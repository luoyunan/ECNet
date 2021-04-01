import collections
from io import StringIO
import pandas as pd

'''
Amino acide encoding modified from 
https://github.com/openvax/mhcflurry/blob/74b751e6d72605eef4a49641d364066193541b5a/mhcflurry/amino_acid.py
'''
COMMON_AMINO_ACIDS_INDEX = collections.OrderedDict(    
    {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
     'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 
     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
     'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20})
AMINO_ACIDS = list(COMMON_AMINO_ACIDS_INDEX.keys())

AMINO_ACID_INDEX = collections.OrderedDict(
    {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
     'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 
     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
     'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,      
     'X': 20, 'Z': 20, 'B': 20, 'J': 20, '-': 20})

'''
CCMPred index of amino acid
https://github.com/soedinglab/CCMpred/blob/2b2f9a0747a5e53035c33636d430f2f11dc186dd/src/sequence.c
'''
CCMPRED_AMINO_ACID_INDEX = collections.OrderedDict(
    {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
     'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
     'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
     'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, '-': 20})
CCMPRED_AMINO_ACIDS = list(CCMPRED_AMINO_ACID_INDEX.keys())

BLOSUM62_MATRIX = pd.read_csv(StringIO("""
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  -
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3  0
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  0
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  0
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1  0
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  0
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3  0
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3  0
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1  0
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1  0
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1  0
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2  0
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3  0
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1  0
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4  0
-  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
"""), sep='\s+').loc[AMINO_ACIDS, AMINO_ACIDS]

ENCODING_DATA_FRAMES = {
    "BLOSUM62": BLOSUM62_MATRIX,
    "one-hot": pd.DataFrame([
        [1 if i == j else 0 for i in range(len(AMINO_ACIDS))]
        for j in range(len(AMINO_ACIDS))
    ], index=AMINO_ACIDS, columns=AMINO_ACIDS)
}