import unicodedata;


def get_stats(ids, counts=None):
    """Given a list of integer, counts the consecutive pairs"""
    counts = {} if counts == None else counts
    for pair in zip(ids[:], ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """In the list of integers(ids), replace of the consecutive pairs of integer with idx"""
    newIds = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newIds.append(idx)
            i += 2
        else:
            newIds.append(ids[i])
            i += 1
    return newIds

class Tokenizer():
    def __init__(self):
        self.merges = {} #({(p0, p1) => idx})
        self.special_tokens = {}
        self.vocab = self._build_vocab()
    
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def train():
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, text):
        raise NotImplementedError

