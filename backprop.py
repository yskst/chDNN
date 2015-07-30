#!/usr/bin/python

"""Training RBM.
Usage: 
  trainrbm.py [options] <nn> <file>
  trainrbm.py -h | --help
options:
   -h, --help    Show this help.
   --gpu=<NUM>   The id of GPU. If -1, processing on CPU. [default: 0]
   --of=<file>   output file name.(npz file)
   --df=<str>    sample data format flags.The flag's detail is follow.
   --mb=<num>    mini-batch size.
   -e <num> --epoch=<num>   the number of ephoch.
   --lr <val>    learning rate [default: 0] 
   --mm <val>    momentum [default: 0] 
   --re <val>    regulalizer. [default: 0]
   ---dr <val>   The probability of dropout. If 0, do not applying dropout. [default: 0]
   --rt <bb|rb>  bb=bernoulli-bernoulli, gb=gaussian-bernoulli.
   --seed <NUM>  The seed of random value. [default: 1234]
""" 

import sys, time
from docopt import docopt

import numpy as np
from chainer import cuda, function, Variable
import chainer.functions as F

import util,dataio

def _str2actf(s):
    sl = s.lower()
    if   sl == "sigmoid": return F.sigmoid
    elif sl == "softmax": return F.softmax
    elif sl == "relu":    return F.relu
    elif sl == "tanh":    return F.tanh
    else:
        panic("Unknown activate function name %s \n" % s)

def _load_nn(fileobj):
    d = np.load(fileobj)
    i = 0
    actfs = []
    params = ()
    while 'w_'+str(i) in d.keys():
        w = d['w_'+str(i)]
        params['l'+str(i)] = F.Linear(w.shape[0], w.shape[1], initialW=w,
                                      initial_bias=d['hbias_'+str(i)])


if __name__=='__main__':
    args = docopt('__doc__', argv=sys.argv[1:])

    outf    = args['--of']
    rbmtype = args['--rt']
    gpu     = int(args['--gpu'])
    mbsize  = int(args['--mb'])
    epoch   = int(args['--epoch'])
    lr      = float(args['--lr'])
    mm      = float(args['--mm'])
    re      = float(args['--re'])
    seed    = int(args['--seed'])
    nn      = args['<nn>']
    trainf  = args['<file>']



