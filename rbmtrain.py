#!/usr/bin/python

"""Training RBM.
Usage: 
  trainrbm.py [options] <visnum> <hidnum> <file>
  trainrbm.py -h | --help
options:
   -h, --help   Show this help.
   --cpu        Processing without GPU.
   --of=<file>  output file name.(npz file)
   --df=<str>   sample data format flags.The flag's detail is follow.
   --mb=<num>   mini-batch size.
   -e <num> --epoch=<num>   the number of ephoch.
   --lr <val>   learning rate [default: 0] 
   --mm <val>   momentum [default: 0] 
   --re <val>   regulalizer. [default: 0]
   --rt <bb|rb> bb=bernoulli-bernoulli, gb=gaussian-bernoulli.
   --seed <NUM> The seed of random value.[default=1234]
""" 

import sys
from docopt import docopt

import numpy as np
from chainer import cuda, Variable, FunctionSet,optimizers
import chainer.functions as F

import util,dataio


def forward_gb(x, model, f):
    """ forwarding of Gaussian-Bernoulli RBM 
        param x:     input data.
        param model: functionset.
        param f:     activate function."""

    x = Variable(x)
    h1 = f(model.v0(x))

if __name__=='__main__':
    args = docopt(__doc__+dataio.get_flags_doc(), argv=sys.argv[1:])
    
    mbsize = int(args['--mb'])
    epoch = int(args['--epoch'])
    lr = flaot(args['--lr'])
    mm = float(args['-mm'])
    re = float(args['--re'])
    seed = int(args['--seed'])
    rbmtype = args['--rt']
    visnum = int(args['<visnum>'])
    hidnum = int(args['<hidnum>'])
    run_with_gpu = not bool(args['--cpu'])

    if rbmtype!="gb" and rbmtype!="bb":
        stderr("Unknown RBM type: %s" % rbmtype)

    data = dataio.dataio(args[<file>], args['--df'], visnum)
    
    # Weight is initialized to random value of standard nomal distribution.
    # Bias is initialized to zeros by default.
    init_w = np.random.normal(0, 1, hidnum, visnum).astype(np.float32)
    model = FunctionSet(v0=F.Linear(visnum, hidnum, initialW=init_w)
                        h0=F.Linear(hidnum, visnum, initialW=init_w.T))
    

