import random
import re

import torch

from vocab import Vocab


TOKEN_PATTERN = re.compile(r"[']*[^'.,?! ]+|[.,?!]")


def tokenize(sent):
    
    if sent == '<SILENCE>':
        return [sent]
    
    return re.findall(TOKEN_PATTERN, sent)


def dialog_reader(path):
    """Reads dialogs that are given in Facebook bAbI dialog format."""

    dialogs = []
    
    with open(path) as f:
        for line in f:
            
            if line in ['\n', '\r\n']:
                
                yield dialogs
                
                dialogs = []
            
            elif '\t' in line:
                
                match = re.search("^\d+ ([^\t]+)\t(.+$)", line)
                if not match:

                    raise ValueError("Invalid dataset format.")
                
                if match[2] == '<SILENCE>':
                    raise ValueError("Invalid dataset format: Bot never keeps silence.")
                
                dialogs.append((match[1], match[2]))
            
            else:               
                dialogs.append((line.split(' ', 1)[1][:-1], None))
    
    if len(dialogs) > 0:
        yield dialogs


def build_dialog_vocab(dialog_dataset_path, candidates_path, time_features=1000):
    """
    Builds two vocabularies. One contains all dialog words along with some special tokens and the second contains
    candidate responses, where the word is a whole sentence.

    :param dialog_dataset_path: Path to the dialog dataset (must be in Facebook bAbI dialog format)
    :param candidates_path: Path to the file containing candidate responses.
    :param time_features: Number of time features to add to the dialog vocabulary.
    :return: tuple containing dialog and candidate response vocabularies, respectively.
    """
    
    vocab = Vocab()
    
    # PAD token index must be zero so we add it first.
    vocab.add_special_token('<pad>')
    
    vocab.add_special_token('<unk>')
    
    vocab.add_special_token('<bot>')
    vocab.add_special_token('<user>')
    
    # adding time features into the vocabulary
    for i in range(time_features):
        vocab.add_special_token('<ts_%d>' % i)
    
    # adding user spoken words to the vocabulary 
    for dialog in dialog_reader(dialog_dataset_path):
        
        for user_utter, _ in dialog:
            
            for word in tokenize(user_utter):
                vocab.add_word(word)
    
    candidate_vocab = Vocab()
    with open(candidates_path) as f:
        for line in f:
            
            sent = line[2:-1]
            
            candidate_vocab.add_word(sent)
            
            for word in tokenize(sent):
                vocab.add_word(word)
                
    vocab.make()
    candidate_vocab.make()
    
    return vocab, candidate_vocab


def sent2vec(vocab, sent):
    """Returns vector representation of a sentence by substituting each word with its index from the vocabulary."""
    
    vec = []
    
    for word in tokenize(sent):
        
        idx = vocab.word_to_index(word)
        
        if idx == -1:
            idx = vocab.word_to_index('<unk>')
        
        vec.append(idx)
    
    return vec


def vec2sent(vocab, vec):
    """Returns original sentence from its vector representation."""
    
    return ' '.join(vocab.index_to_word(idx) for idx in vec)


class DialogReader:
    """
    Represents an iterator over the mini-batches of data samples for training/evaluation.

    Single data sample is a triple containing the memory (current dialog history), query (current user utterance)
    and the label (ground truth bot response), respectively.

    When dealing with mini-batches, data samples are sorted by memory length in advance, so that mini-batches are
    approximately same size for computation efficiency.
    """

    def __init__(self, 
                 dialog_data_path,
                 dialog_vocab,
                 candidate_vocab,
                 max_memory_size,
                 batch_size,
                 drop_last_batch=False,
                 shuffle_data=False,
                 eval_mode=False):
        """
        :param dialog_data_path: Path to the dialog dataset.
        :param dialog_vocab: The dialog vocabulary (word level).
        :param candidate_vocab: The vocabulary of candidate responses (sent. level).
        :param max_memory_size: The maximum size of the dialog history. If exceeded, the earliest utterances are dropped.
        :param batch_size: The size of mini-batch.
        :param drop_last_batch: If the number of data samples isn't divisible by batch_size, the last smaller mini-batch is dropped.
        :param shuffle_data: Shuffle mini-batches before returning the iterator.
        :param eval_mode: If true, every mini-batch has size 1 (regardless batch_size) and comes with an unique dialog id,
        so that mini-batches from the same dialog have same ids. Useful when evaluating per dialog accuracy.
        """

        self._dialog_data_path = dialog_data_path
        self._dialog_vocab = dialog_vocab
        self._candidate_vocab = candidate_vocab
        self._max_memory_size = max_memory_size
        self._batch_size = batch_size if not eval_mode else 1    
        self._drop_last_batch = drop_last_batch
        self._shuffle_data = shuffle_data

        # In eval mode batch_size is automatically set to 1, dataset isn't sorted/shuffled, batch comes with dialog id. 
        self._eval_mode = eval_mode
        
        self._load_data()
        
        if not eval_mode:
            self._dataset.sort(key=lambda x: len(x[0]))
        
        self._batches = []
        
        batch = []
        for sample in self._dataset:
            
            batch.append(sample)
            
            if len(batch) == self._batch_size:

                self._add_batch(batch)

                batch = []
                
        if len(batch) > 0 and not self._drop_last_batch:
            self._add_batch(batch)

    def _add_batch(self, batch):
        
        if self._eval_mode:
            self._batches.append((batch[0][0], self._batch_to_tensor([batch[0][1]])))
        else:
            self._batches.append(self._batch_to_tensor(batch))

    def _load_data(self):
        
        # Vectorizing candidate responses.
               
        candidate_vec_max_len = 0
        candidate_vecs = []
        for i in range(len(self._candidate_vocab)):
            
            sent = self._candidate_vocab.index_to_word(i)
            
            candidate_vec = [self._dialog_vocab.word_to_index(w) for w in tokenize(sent)]
            
            candidate_vec_max_len = max(candidate_vec_max_len, len(candidate_vec))
            
            candidate_vecs.append(candidate_vec)

        # Creating tensor of (num_candidates, max_candidate_len) size to store all candidate responses.
        
        self._candidate_vecs = torch.LongTensor(len(candidate_vecs), candidate_vec_max_len).fill_(self._dialog_vocab.word_to_index('<pad>'))
        
        for i in range(len(candidate_vecs)):
            self._candidate_vecs[i,:len(candidate_vecs[i])] = torch.LongTensor(candidate_vecs[i])
        
        # Building dialog dataset containing (current_meomry, query, label) triples.
        
        self._dataset = []
        for dialog_i, dialog in enumerate(dialog_reader(self._dialog_data_path)):
            
            user_utters, bot_utters = zip(*dialog)
            
            i, tm = 0, 0
            memories = []
            while i < len(dialog):
                
                if bot_utters[i]:
                
                    query = sent2vec(self._dialog_vocab, user_utters[i])

                    label = self._candidate_vocab.word_to_index(bot_utters[i])
                    
                    if self._eval_mode:
                        self._dataset.append((dialog_i, (memories[:], query, label)))
                    else:    
                        self._dataset.append((memories[:], query, label))

                    self._write_memory(memories, query, tm, 0)
                    self._write_memory(memories, sent2vec(self._dialog_vocab, bot_utters[i]), tm + 1, 1)
                    
                    i, tm = i + 1, tm + 2
                
                # Handling 'displaying options' case.
                else:
                    
                    while not bot_utters[i]:
                        
                        self._write_memory(memories, sent2vec(self._dialog_vocab, user_utters[i]), tm, 0)
                        
                        i, tm = i + 1, tm + 1

    def _write_memory(self, memories, memory, time, speaker_id):
        
        memory = self._add_speaker_feature(memory, speaker_id)
        memory = self._add_time_feature(memory, time)
        
        if len(memories) == self._max_memory_size and self._max_memory_size > 0:
            del memories[0]
            
        if self._max_memory_size > 0:
            memories.append(memory)

    def _add_speaker_feature(self, vec, speaker_id):
        
        return [self._dialog_vocab.word_to_index(['<user>', '<bot>'][speaker_id])] + vec

    def _add_time_feature(self, vec, time):
              
        return [self._dialog_vocab.word_to_index('<ts_%d>' % time)] + vec

    def _batch_to_tensor(self, batch):
        
        pad = self._dialog_vocab.word_to_index('<pad>')
        
        memories, queries, labels = zip(*batch)
        
        batch_size = len(batch)
        max_mem_len = max(1, 1, *[len(m) for m in memories])
        max_vec_len = max(1, 1, *[len(v) for m in memories for v in m])
        max_query_len = max(len(q) for q in queries)

        mem_tensor = torch.LongTensor(batch_size, max_mem_len, max_vec_len).fill_(pad)
        
        query_tensor = torch.LongTensor(batch_size, max_query_len).fill_(pad)
        
        label_tensor = torch.stack([torch.LongTensor([label]) for label in labels])
        
        for i in range(batch_size):
            for j in range(len(memories[i])):
                mem_tensor[i,j,:len(memories[i][j])] = torch.LongTensor(memories[i][j])
                
            query_tensor[i,:len(queries[i])] = torch.LongTensor(queries[i])
        
        return mem_tensor, query_tensor, label_tensor

    def __iter__(self):

        if not self._eval_mode and self._shuffle_data:
            random.shuffle(self._batches)
        
        return iter(self._batches)

    def __len__(self):
        
        return len(self._batches)
