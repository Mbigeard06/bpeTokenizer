from base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        #input processing
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) #list of integer in range 0..255

        merges = {}
        #associate bytes with its corresponding index
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            #Get the most common pair in the ids
            pair = max(stats, key = stats.get)
            #Save the merge
            idx = 256 + i
            merges[pair] = idx
            #Update ids
            ids = merge(ids, pair, idx)
            #Update the vocab 
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] #get the byte associate with the index
        #Update class variables
        self.merges = merges
        self.vocab = vocab
    
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        #Get stats
        while len(ids) > 2:
            stats = get_stats(ids)
            #get the existing pair that has the lowest index in merge 
            pair = min(stats, key= lambda p: self.merges.get(p, float("inf")))
            #nothing to merge
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids


