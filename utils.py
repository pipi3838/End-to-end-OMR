import Levenshtein as Lev
import torch
from six.moves import xrange

class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.
    Arguments:
        labels (string): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=0):
        #labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ#"
        self.labels = labels
        self.int_to_char = dict([(i+1, c) for (i, c) in enumerate(labels)])
        self.int_to_char[0] = ' '  #加入blank
        self.blank_index = blank_index
        space_index = len(self.int_to_char)  #1782(包含blank): To prevent errors in decode, we add an out of bounds index for the space

class GreedyDecoder():
    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False):
        strings = []
        for x in xrange(len(sequences)): #一個data
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            sequence = sequences[x].cpu().numpy()
            string = self.process_string(sequence, seq_len, remove_repetitions)
            strings.append(string)  # We only return one path
        return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = []
        for i in range(sequence.shape[0]):
            char_int = sequence[i]
            if char_int != 0:  #非空白
                if remove_repetitions and i != 0 and char_int == sequence[i-1]:
                    pass
                else:
                    string.append(char_int)
        get_len = min(size,len(string))
        return string[:get_len]

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.
        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True)
        return strings, max_probs
        
def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

    