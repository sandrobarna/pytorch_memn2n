import argparse
import os

import torch
from torch.autograd import Variable

from data_utils import DialogReader
from model import MemN2N
from vocab import Vocab


def calc_accuracy_per_response(model, data_reader, use_cuda):
    """
    Calculates per response accuracy, that is, the ratio of correct responses out of all responses.

    :param model: Trained model used for prediction.
    :param data_reader: DialogReader instance that provides iterator over samples.
    :param use_cuda: If True, calculations will be performed on GPU.
    """

    n_correct = 0
    samples_total = 0
    
    for i_batch, sample_batched in enumerate(data_reader):
            
            sample, query, label = sample_batched

            if use_cuda:
                sample = sample.cuda()
                query = query.cuda()
                label = label.cuda()
        
            pred, _ = model(Variable(sample), Variable(query))
             
            n_correct += torch.sum(torch.max(pred.data, 1)[1] == label.squeeze(1))
            samples_total += pred.size()[0]

    return n_correct / samples_total


def calc_accuracy_per_dialog(model, data_reader):
    """
    Calculates per dialog accuracy, that is, the ratio of dialogs where every response is correct out of all dialogs.

    :param model: Trained model used for prediction.
    :param data_reader: DialogReader instance that provides iterator over samples.
    """
        
    acc = dict()
    
    for dialog_i, sample in data_reader:
            
        memory, query, label = sample
        
        pred, _ = model(Variable(memory), Variable(query))
             
        pred = (torch.max(pred.data, 1)[1] == label.squeeze(1))[0]
            
        if dialog_i in acc:
            acc[dialog_i][0] += pred
            acc[dialog_i][1] += 1
        else: 
            acc[dialog_i] = [pred, 1]

    return sum(1 if i[0] == i[1] else 0 for i in acc.values()) / len(acc)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Goal-Oriented Chatbot using End-to-End Memory Networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model_dir', type=str, help='trained model path')
    parser.add_argument('test_path', type=str, help='test data path')  
    
    parser.add_argument('--maxmemsize', type=int, metavar='N', default=100, help='memory capacity')

    args = parser.parse_args()

    # loading vocabularies and the trained model 

    dialog_vocab = Vocab.load(os.path.join(args.model_dir, 'dialog_vocab'))
    candidates_vocab = Vocab.load(os.path.join(args.model_dir, 'candidates_vocab'))

    model = MemN2N.load(os.path.join(args.model_dir, 'model'))

    test_data_reader_per_resp = DialogReader(args.test_path, dialog_vocab, candidates_vocab, args.maxmemsize, 1, False, False, False)
    test_data_reader_per_dial = DialogReader(args.test_path, dialog_vocab, candidates_vocab, args.maxmemsize, 1, False, False, True)
    
    print("Per Response Accuracy: ", calc_accuracy_per_response(model, test_data_reader_per_resp, False))
    print("Per Dialog Accuracy: ", calc_accuracy_per_dialog(model, test_data_reader_per_dial))
