import pickle
from operator import itemgetter


class Vocab:
    """Represents a vocabulary for storing words and their indices and provides mapping between them."""

    def __init__(self):
        
        self._words = dict()
        self._spec_tokens = []

    def add_special_token(self, token):
        
        self._spec_tokens.append(token)

    def add_word(self, word):
        
        self._words[word] = self._words.get(word, 0) + 1

    def make(self, n_words=-1):
        """
        Builds mapping between words and their indices and vice versa. Should be called before using vocabulary.
        :param n_words: Number of top frequent words to be included in final vocabulary.
        If negative, all words are used.
        """
        
        self._ensure_validity()
        
        vocab_size = len(self._spec_tokens) + len(self._words)
        
        self._vocab_size = min(n_words, vocab_size) if n_words > -1 else vocab_size 

        # Spec tokens must preserve the order they have been added
        # Words are ordered by decreasing frequency (stable sort)
        self._idx2word = self._spec_tokens + [k for k, v in sorted(self._words.items(), key=itemgetter(1), reverse=True)[:self._vocab_size]]
        
        self._word2idx = dict((v, k) for k, v in enumerate(self._idx2word))

    def word_to_index(self, word):
        
        return self._word2idx.get(word, -1)

    def index_to_word(self, index):
        
        return self._idx2word[index] if 0 <= index < self._vocab_size else None

    def _ensure_validity(self):
        
        unique_specials = set(self._spec_tokens)
        
        if len(unique_specials) != len(self._spec_tokens):
            raise ValueError("Single spec token was added more than once.")
            
        if len(unique_specials & self._words.keys()) != 0:
            raise ValueError("Spec tokens and words mustn't have common elements.")
        
    def __len__(self):
        
        return self._vocab_size

    @staticmethod
    def save(vocab, path):
        
        with open(path, 'wb') as f:
            pickle.dump(vocab, f)
        
    @staticmethod    
    def load(path):
        
        with open(path, 'rb') as f:
            return pickle.load(f)