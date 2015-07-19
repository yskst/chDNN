#!/usr/bin/python

"""Training RBM.
Usage: 
  trainrbm.py [options] <visnum> <hidnum> <file>
  trainrbm.py -h | --help
options:
   -h, --help   Show this help.
   --gpu=<NUM>  The id of GPU.If 0, processing on CPU.[ default: 1]
   --of=<file>  output file name.(npz file)
   --df=<str>   sample data format flags.The flag's detail is follow.
   --mb=<num>   mini-batch size.
   -e <num> --epoch=<num>   the number of ephoch.
   --lr <val>   learning rate [default: 0] 
   --mm <val>   momentum [default: 0] 
   --re <val>   regulalizer. [default: 0]
   --rt <bb|rb> bb=bernoulli-bernoulli, gb=gaussian-bernoulli.
   --seed <NUM> The seed of random value.[default: 1234]
""" 

import sys
from docopt import docopt

import numpy as np
from chainer import cuda, Variable, FunctionSet
import chainer.functions as F

import util,dataio


class train_cpu:
    def init(self, visnum, hidnum, seed, f=F.sigmoid, bb=true):
        np.random.seed(seed)
        self.__f__ = f
        self.__model__ = F1.Linear(visnum, hidnum)
        self.__vbias__ = np.zeros(visnum, dtype=np.float32)
        self.__gvbias__= np.empty_like(self.__vbias__)
        if bb:
            self.__inverse__ = self.__inverse_bb
        else:
            self.__inverse__ = self.__inverse_gb
    
    def __inverse_bb(x):
        return self.__f__(x * self.__model__.W.T + self.__vbias__)
    def __inverse_gb(x):
        return x * self.__model__.W.T + self.__vbias__

    def __sampling(self, p):
    """ Samping hidden layer's neuron state from probability. """
        return np.random.binomial(1, p=p, size=self.__model__.shape)

    def trainging(self, x, lr, mm, re):
        h0act = self.__f__(self.__model__.forward_cpu(x))
        h0smp = self.sampling(h1act)
        v1act = self.__inverse__(h1smp)
        h1act = self.__f__(self.__model__.forward_cpu(x))
        
        # Calcurate gradient of each parameter.
        gw = v1act.T*h1act - v0act.T*h0act
        gb = np.sum(h1act,axis=0) - np.sum(h0act,axis=0)
        gvb= np.sum(v1act,axis=0) - np.sum(v0act,axis=0)

        # Calcurate difference for update.
        self.__model__.gw= -lr*gw  +mm*self.__model__.gw -re*self.__model__.W
        self.__model__.gb= -lr*gb  +mm*self.__model__.gb
        self.__gvbias__  = -lr*gvb +mm*self.__gvbias__
        
        # Update each parameter.
        self.__model__.W += self.__model__.gw
        self.__model__.b += self.__model__.gb
        self.__vbias__   += self.__gvbias__


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
    

