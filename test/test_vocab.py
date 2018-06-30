import unittest

from vocab import Vocab


class VocabTestCase(unittest.TestCase):
    
    def setUp(self):
        
        self._vocab = Vocab()
        
        self._spec_tokens = ['<pad>', '<unk>']
        self._words = ['Hello', 'world', 'a', 'HELLO WORLD !!!', '2018', ':)']
        
        for w in self._spec_tokens:
            self._vocab.add_special_token(w)
            
        for w in self._words:
            self._vocab.add_word(w)

    def test_len(self):
            
        self._vocab.make()
        
        self.assertEqual(len(self._vocab), len(self._spec_tokens) + len(self._words))
        
    def test_word2index_index2word(self):
            
        self._vocab.make()
        
        for i, w in enumerate(self._spec_tokens + self._words):
            self.assertEqual(self._vocab.word_to_index(w), i)
            self.assertEqual(self._vocab.index_to_word(i), w)
        
        self.assertEqual(self._vocab.word_to_index('Unknown word'), -1)
        
        self.assertIsNone(self._vocab.index_to_word(-1))
        self.assertIsNone(self._vocab.index_to_word(297196412))
        
    def test_limited_vocab_size(self):
        
        for w in ['frequent word', 'frequent word']:
            self._vocab.add_word(w)
            
        n_words = 5
            
        self._vocab.make(n_words)
        
        self.assertEqual(len(self._vocab), n_words)
        
        for i, w in enumerate(self._spec_tokens + ['frequent word'] + self._words):
            self.assertEqual(self._vocab.word_to_index(w), i)
            self.assertEqual(self._vocab.index_to_word(i), w)
            
            if i == n_words - 1:
                break
                
    def test_spec_token_occurring_in_words(self):
        
        self._vocab.add_special_token('common word')
        self._vocab.add_word('common word')

        with self.assertRaises(ValueError):
            self._vocab.make()
            
    def test_spec_token_added_twice(self):
        
        self._vocab.add_special_token('token')
        self._vocab.add_special_token('token')

        with self.assertRaises(ValueError):
            self._vocab.make()