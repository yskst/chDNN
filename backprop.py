#!/usr/bin/python

"""Training RBM.
Usage: 
  trainrbm.py [options] <file> <tar>
  trainrbm.py -h | --help
options:
   -h, --help    Show this help.
   --gpu=<NUM>   The id of GPU. If -1, processing on CPU. [default: 0]
   --mlp=<FILE>   The initial values of MLP parameter.
   --of=<file>   output file name.(npz file)
   --df=<str>    sample data format flags.The flag's detail is follow.
   --tf=<str>    the target data format flags.
   --ot=<c|f>    The target type is feature or category. 
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
from chainer import cuda, function, Variable, Functionset, optimizers
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
        s = str(i)
        w = d['w_'+s]
        params['l'+s] = F.Linear(w.shape[0], w.shape[1], initialW=w,
                                      initial_bias=d['hbias_'+s])
        actf = _str2actf(d['type_'+s])
        actfs.append(actf)
    model = Functionset(**params)
    return model, actfs

def forward_mse(x_data, y_data, model, actf):
    x = Variable(x_data)
    y = Variable(y_data)
    h = actf[0](model.l0(x))
    for i in range(1, len(actf)):
        m = getattr(model, 'l'+str(i))
        h = actf[i](m(h))
    return F.mean_squared_error(h, y)

def forward_cross_entoropy(x_data, y_data, model, actf):
    x = Variable(x_data)
    y = Variable(y_data)
    h = actf[0](model.l0(x))
    l = len(actf)
    for i in range(1, l):
        m = getattr(model, 'l'+str(i))
        if i < l-1:
            h = actf[i](m(h))
        else:
            h = m(h)
    return F.softmax_cross_entropy(h, y), F.accuracy(h,y)

if __name__=='__main__':
    args = docopt('__doc__'+dataio.get_flags_doc(), argv=sys.argv[1:])

    outform = args['--of']
    tarform = args['--tf']
    otype   = args['--ot']
    rbmtype = args['--rt']
    nn      = args['--mlp']
    gpu     = int(args['--gpu'])
    mbsize  = int(args['--mb'])
    epoch   = int(args['--epoch'])
    lr      = float(args['--lr'])
    mm      = float(args['--mm'])
    re      = float(args['--re'])
    seed    = int(args['--seed'])
    tar     = args['<tar>']
    trainf  = args['<file>']


    model, actfs = _load_nn(nn)
    optimizer = opimizers.MomentumSGD(lr=lr,momentum=mm)
    optimizer.setup(model.collect_parameters())
    optimizer.zero_grads()
