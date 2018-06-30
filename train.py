import argparse
import os
import time

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from data_utils import DialogReader
from data_utils import build_dialog_vocab
from eval import calc_accuracy_per_response
from model import MemN2N
from vocab import Vocab


def to_time(seconds):
    
    seconds = int(seconds)
    h = seconds / 3600
    m = seconds / 60 % 60
    seconds %= 60
    return '%dh %dm %ds' % (h, m, seconds)


def train_step(model, optimizer, loss_criterion, max_grad_norm, memory, query, target, use_cuda):
    
    optimizer.zero_grad()

    memory = memory.cuda() if use_cuda else memory

    query = query.cuda() if use_cuda else query

    target = Variable(target).squeeze(1)
    
    pred, _ = model(Variable(memory), Variable(query))
    
    loss = loss_criterion(pred.cpu(), target)
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
    
    optimizer.step()
    
    return loss.data[0]


def train(model, 
    data_loader, 
    dev_set_reader, 
    n_epochs, 
    lr, 
    decay_factor, 
    decay_every,
    max_grad_norm, 
    print_interval, 
    summary_interval, 
    use_cuda):
    
    start_ts = time.time()
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    lr_scheduler = StepLR(optimizer, step_size=decay_every, gamma=decay_factor)
    
    loss_criterion = nn.NLLLoss()
    loss_criterion = loss_criterion.cuda() if use_cuda else loss_criterion
    
    loss_total_print = 0
    loss_total_summary = 0
    losses = []
    
    i = 0
    for i_epoch in range(n_epochs):
        
        for i_batch, sample_batched in enumerate(data_loader):
            
            sample, query, label = sample_batched
            
            loss = train_step(model, optimizer, loss_criterion, max_grad_norm, sample, query, label, use_cuda)
            
            loss_total_print += loss
            loss_total_summary += loss
            losses.append(loss)
            
            if i % print_interval == 0:
                print("Epoch: %d, Iter: %d, Loss: %.5f" % (i_epoch + 1, i_batch + 1, loss_total_print / print_interval))
                loss_total_print = 0
                
            if (i + 1) % summary_interval == 0:
                
                avg_loss = loss_total_summary / summary_interval
                loss_total_summary = 0
                
                dev_acc = calc_accuracy_per_response(model, dev_set_reader, use_cuda)

                print('\n---------- SUMARRY ----------')
                print("Time elapsed: %s, Train Loss: %.5f, Dev Accuracy: %.5f\n" % (to_time(time.time() - start_ts), avg_loss, dev_acc))
            
            i += 1
            
        lr_scheduler.step()
    
    return losses


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Goal-Oriented Chatbot using End-to-End Memory Networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('train_path', type=str, help='training data path')
    parser.add_argument('dev_path', type=str, help='development data path')  
    parser.add_argument('candidates_path', type=str, help='candidate responses data path')  

    gr_model = parser.add_argument_group('Model Parameters')
    gr_model.add_argument('--edim', type=int, metavar='N', default=32, help='internal state dimension')
    gr_model.add_argument('--nhops', type=int, metavar='N', default=1, help='number of memory hops')
    gr_model.add_argument('--init_std', type=float, metavar='N', default=0.1, help='weight initialization std')

    gr_train = parser.add_argument_group('Training Parameters')
    gr_train.add_argument('--gpu', action="store_true", default=False, help='use GPU for training')
    gr_train.add_argument('--lr', type=float, metavar='N', default=0.01, help='initial learning rate')
    gr_train.add_argument('--decay_factor', type=float, default=0.5, help='learning rate decay factor')
    gr_train.add_argument('--decay_every', type=int, default=25, help='# of epochs learning rate is changed')
    gr_train.add_argument('--batchsize', type=int, metavar='N', default=32, help='minibatch size')
    gr_train.add_argument('--epochs', type=int, metavar='N', default=50, help='initial learning rate')
    gr_train.add_argument('--maxgradnorm', type=int, metavar='N', default=40, help='maximum gradient norm')
    gr_train.add_argument('--maxmemsize', type=int, metavar='N', default=100, help='memory capacity')
    gr_train.add_argument('--shuffle', action="store_true", default=True, help='shuffle batches before every epoch')
    gr_train.add_argument('--save_dir', type=str, default=None, help='path to save the model')

    args = parser.parse_args()

    # build data, initialize model and start training.

    dialog_vocab, candidates_vocab = build_dialog_vocab(args.train_path, args.candidates_path, 1000)    

    trn_data_reader = DialogReader(args.train_path, dialog_vocab, candidates_vocab, args.maxmemsize, args.batchsize, False, args.shuffle, False)
    dev_data_reader = DialogReader(args.dev_path, dialog_vocab, candidates_vocab, args.maxmemsize, args.batchsize, False, False, False)

    candidate_vecs = Variable(trn_data_reader._candidate_vecs)
    candidate_vecs = candidate_vecs.cuda() if args.gpu else candidate_vecs

    model = MemN2N(args.edim, len(trn_data_reader._dialog_vocab), candidate_vecs, args.nhops, args.init_std)

    if args.gpu:
        model.cuda()

    train(model, trn_data_reader, dev_data_reader, args.epochs, args.lr, args.decay_factor, args.decay_every, args.maxgradnorm, 50, 500, args.gpu)
    
    # saving trained model and vocabularies.

    save_dir = args.save_dir
    if not save_dir:
        save_dir = os.getcwd()

    save_dir = os.path.join(save_dir, 'model_' + str(time.time()))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        raise ValueError("Model save path already exists")

    Vocab.save(dialog_vocab, os.path.join(save_dir, 'dialog_vocab'))
    Vocab.save(candidates_vocab, os.path.join(save_dir, 'candidates_vocab'))
    model.save(os.path.join(save_dir, 'model'))
