# End-to-End Goal-Oriented Dialog

This repo contains a PyTorch implementation of the End-to-End Memory Network as described in the paper *[Learning end-to-end goal-oriented dialog.](https://arxiv.org/pdf/1605.07683.pdf)* Also there is a code for replicating the results on T1-T5 bAbI tasks and a jupyter notebook file for visualizing memory attentions of a learned model. 


### Requirements

```
- python 3.6
- pytorch 0.3.0
```

### Running

First you need to download [bAbI dialog dataset](https://fb-public.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w).

To run the training, use the following pattern:
```
python train.py /path/to/dataset/train_set_file.txt /path/to/dataset/dev_set_file.txt /path/to/dataset/candidates_file.txt
```
There are different command line arguments for adjusting model and training parameters. For complete list, run
```
python train.py -h
```
For evaluation, use:
```
python eval.py /path/to/saved/model/dir /path/to/dataset/test_set_file.txt
```
