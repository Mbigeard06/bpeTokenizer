import regex as re
from base import Tokenizer, get_stats, merge

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern = None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        #split the text into chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256) } 
        for i in range(num_merges):
            stats = {}
            #check pairs score on all the chunks
            for chunks_ids in ids:
                get_stats(chunks_ids, stats)
            #get the pair with the highest count 
            pair = max(stats, key=stats.get)
            #merge id
            idx = 256 + i
            #merge the pair in all chunks 
            ids = [merge(chunks_ids, pair, idx) for chunks_ids in ids]
            #save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges = merges
        self.vocab = vocab
    
    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            #get tokens value (idx => byte)
            part_bytes.append(self.vocab[idx])
        #concatenate
        text_bytes = b"".join(part_bytes)
        #decode the whole chain
        text = text_bytes.decode("utf-8", errors='replace')
        return text
    
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        #while we can merge
        while len(ids) > 2:
            stats = get_stats(ids)
            #get the lowest indexed merged pair that appears in text_bytes
            pair = min(stats, key = lambda p : self.merges.get(p, float("inf")))
            #there is no merge to do
            if pair not in self.merges:
                break
            #merge the tokens 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(self, text):
        #get the different chunk thanks to the regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            #encode the chunk
            text_bytes = chunk.encode("utf-8")
            #merge the tokens 
            chunks_ids =  self._encode_chunk(text_bytes)
            ids.extend(chunks_ids)
        return ids





