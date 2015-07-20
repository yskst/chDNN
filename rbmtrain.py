#!/usr/bin/python

"""Training RBM.
Usage: 
  trainrbm.py [options] <visnum> <hidnum> <file>
  trainrbm.py -h | --help
options:
   -h, --help   Show this help.
   --gpu=<NUM>  The id of GPU. If -1, processing on CPU.[ default: 0]
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
    def init(self, visnum, hidnum, seed, f=f.sigmoid, bb=true):
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

    def __sampling(self, ndim, p):
    """ Samping hidden layer's neuron state from probability. """
        return np.random.binomial(1, p=p, size=ndim)

    def trainging(self, x, lr, mm, re):
        h0act = self.__f__(self.__model__.forward_cpu(x))
        h0smp = self.sampling(self.__model__.bias.shape, h0act)
        v1act = self.__inverse__(h0smp)
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

        return np.sum((v0act-v1act)**2)

    def get_param(self):
        model = self.__model__
        return mode.W, model.bias, self.__vbias__

class train_gpu(train_cpu):
    def __init__(self,visnum, hidnum, seed, f=f.sigmoid, bb=true, gpuid=0):
        if gpuid >= 0:
            util.panic("GPU ID is out of range(>= 0)")

        gpu = cuda.get_device(gpuid)
        self.gpu = gpu
        cuda.init(gpu)
        # Initialize random number generator.
        self.__randomgen__ = cuda.get_generator(gpu)
        cuda.seed(gpu)
       
        self.__model__ = F1.Linear(visnum, hidnum).to_gpu(gpu)
        self.__vbias__ = cuda.to_gpu(np.zeros(visnum,dtype=np.float32),gpu)
        self.__gvbias__= cuda.to_gpu(np.empty_like(self.__vbias__),    gpu)
        if bb:
            self.__inverse__ = self.__inverse_bb
        else:
            self.__inverse__ = self.__inverse_gb
    
    def __sampling(self, ndim, p):
    """ Samping hidden layer's neuron state from probability. """
        return self.__randomgen__.gen_uniform(ndim, np.float32) > p

    def train(self, x, lr, mm, re):
        x_gpu = Variable(cuda.to_gpu(x, self.gpu))
        
        h0act = self.__f__(self.__model__.forward_gpu(x))
        h0smp = self.sampling(self.__model__.bias.shape, h0act)
        v1act = self.__inverse__(h0smp)
        h1act = self.__f__(self.__model__.forward_gpu(x))
        
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

        return cuda.gpuarray.sum((v0act-v1act)**2)

     def get_param(self):
        model = self.__model__
        return cuda.to_cpu(mode.W), cuda.to_cpu(model.bias), cuda.to_cpu(self.__vbias__)



if __name__=='__main__':
    args = docopt(__doc__+dataio.get_flags_doc(), argv=sys.argv[1:])
    
    gpuid  = int(args('--gpu'))
    mbsize = int(args['--mb'])
    epoch  = int(args['--epoch'])
    lr = flaot(args['--lr'])
    mm = float(args['-mm'])
    re = float(args['--re'])
    seed = int(args['--seed'])
    rbmtype = args['--rt']
    visnum = int(args['<visnum>'])
    hidnum = int(args['<hidnum>'])

    if rbmtype!="gb" and rbmtype!="bb":
        util.stderr("Unknown RBM type: %s" % rbmtype)
    elif rbmtype == "gb":
        bbrbm = false
    else:
        bbrbm = true

    data = dataio.dataio(args[<file>], args['--df'], visnum)
    ndata = data.shape[0]
    
    if gpuid < 0:
        trainer = train_cpu(visnum, hidnum, seed, bb=bbrbm)
    else:
        trainer = trian_gpu(visnum, hidnum, seed, bb=bbrbm, gpuid=gpuid)
        np.random.seed(seed)

    for i in range(epoch):
        e = 0.0
        mblst = np.random.permutation(ndata)

        for mb in range(0, ndata, mbsize):
            e += trainer.train(data[mblst[mb:mb+mbsize]])
        e /= (ndata/mbsize)
        util.stdout("%4d th-epoch mse= %7e\n" % (i, e))
