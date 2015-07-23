#!/usr/bin/python

"""Training RBM.
Usage: 
  trainrbm.py [options] <visnum> <hidnum> <file>
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
   --rt <bb|rb>  bb=bernoulli-bernoulli, gb=gaussian-bernoulli.
   --seed <NUM>  The seed of random value. [default: 1234]
""" 

import sys
from docopt import docopt

import numpy as np
from chainer import cuda, Variable, FunctionSet
import chainer.functions as F

import util,dataio


class train_cpu:
    def __init__(self, visnum, hidnum, seed, f=F.sigmoid, bb=True):
        np.random.seed(seed)
        self.__f__ = f
        self.__model__ = F.Linear(visnum, hidnum)
        self.__vbias__ = np.zeros(visnum, dtype=np.float32)
        self.__model_inv__ = F.Linear(hidnum, visnum, 
                 initialW=self.__model__.W.T.copy())

        self.__model__.zero_grads()
        self.__model_inv__.zero_grads()

        if bb:
            self.__inverse__ = self.__inverse_bb
        else:
            self.__inverse__ = self.__inverse_gb

    def __inverse_bb(self, x):
        return self.__f__(self.__model_inv__(x))
    def __inverse_gb(self, x):
        return self.__model_inv__(x)

    def __sampling(self, p):
        """ Samping hidden layer's neuron state from probability. """
        return np.random.binomial(1, p=p, size=p.shape).astype(np.float32)

    def train(self, x, lr=0.0, mm=0.0, re=0.0):
        ndata = x.shape[0]
        x_cpu = Variable(x)

        h0act = self.__f__(self.__model__(x_cpu))
        h0smp = self.__sampling(h0act.data)
        v1act = self.__inverse__(Variable(h0smp))
        h1act = self.__f__(self.__model__(v1act))
        
        # Calcurate gradient of each parameter.
        gw = F.matmul(v1act,h1act,transa=True).data - F.matmul(x_cpu, h0act, transa=True).data / ndata
        gb = (np.sum(h1act.data, axis=0) - np.sum(h0act.data,axis=0)) / ndata
        gvb= (np.sum(v1act.data, axis=0) - np.sum(x_cpu.data,axis=0)) / ndata

        # Calcurate difference for update.
        self.__model__.gW     = -lr*gw.T  +mm*self.__model__.gW -re*self.__model__.W
        self.__model__.gb     = -lr*gb  +mm*self.__model__.gb
        self.__model_inv__.gb = -lr*gvb +mm*self.__model_inv__.gb
        
        # Update each parameter.
        self.__model__.W += self.__model__.gW
        self.__model__.b += self.__model__.gb
        self.__model_inv__.W    = self.__model__.W.T.copy()
        self.__model_inv__.b   += self.__model_inv__.gb

        return np.mean((x-v1act.data)**2)

    def get_param(self):
        model = self.__model__
        return mode.W, model.bias, self.__vbias__

class train_gpu(train_cpu):
    def __init__(self,visnum, hidnum, seed, f=F.sigmoid, bb=True, gpuid=0):
        if gpuid < 0:
            util.panic("GPU ID is out of range(>= 0)")

        gpu = cuda.get_device(gpuid)
        self.gpu = gpu
        cuda.init(gpuid)
        # Initialize random number generator.
        self.__randomgen__ = cuda.get_generator(gpu)
        cuda.seed(seed, gpu)
       
        self.__f__         = f
        self.__model__     = F.Linear(visnum, hidnum).to_gpu(gpu)
        self.__model_inv__ = F.Linear(hidnum, visnum, 
                  initialW=self.__model__.W.T.copy()).to_gpu(gpu)
        self.__model__.zero_grads()
        self.__model_inv__.zero_grads()

        if bb:
            self.__inverse__ = self.__inverse_bb
        else:
            self.__inverse__ = self.__inverse_gb
    
    def __inverse_bb(self, x):
        return self.__f__(F.linear(x, self.__model__.W.T, self.__vbias__))
    def __inverse_gb(self, x):
        return self.__model_inv__(x)

    def __sampling(self, ndim, p):
        """ Samping hidden layer's neuron state from probability. """
        rnd = self.__randomgen__.gen_uniform(p.shape, np.float32) 
        rnd = p < rnd
        return Variable(rnd)
    
    def __cusum(self, x_variable, axis=None):
        """ The GPUArray versin of numpy.sum"""
        return cuda.cumisc.sum(x_variable.data, axis)

    def train(self, x, lr=0.0, mm=0.0, re=0.0):
        ndata = x.shape[0]
        x_gpu = Variable(cuda.to_gpu(x,self.gpu))
        h0act = self.__f__(self.__model__(x_gpu))
        h0smp = self.__sampling(self.__model__.b.shape, h0act.data)
        v1act = self.__inverse__(h0smp)
        h1act = self.__f__(self.__model__(v1act))
        
        # Calcurate gradient of each parameter.
        gw = F.matmul(v1act,h1act,transa=True).data - F.matmul(x_gpu,h0act, transa=True).data
        gb = self.__cusum(h1act,axis=0) - self.__cusum(h0act,axis=0)
        gvb= self.__cusum(v1act,axis=0) - self.__cusum(x_gpu,axis=0)
        
        gw /= ndata
        gb /= ndata
        gvb /= ndata

        # Calcurate difference for update.
        self.__model__.gW = -lr*gw.T  +mm*self.__model__.gW -re*self.__model__.W
        self.__model__.gb      = -lr*gb  +mm*self.__model__.gb
        self.__model_inv__.gb  = -lr*gvb +mm*self.__model_inv__.gb
        
        # Update each parameter.
        self.__model__.W += self.__model__.gW
        self.__model__.b += self.__model__.gb
        self.__model_inv__.W    = self.__model__.W.T.copy()
        self.__model_inv__.b   += self.__model_inv__.gb

        mse = cuda.gpuarray.sum((x_gpu.data - v1act.data)**2) / ndata / x.shape[1]
        return cuda.to_cpu(mse)


    def get_param(self):
        model = self.__model__
        return cuda.to_cpu(mode.W), cuda.to_cpu(model.bias), cuda.to_cpu(self.__vbias__)



if __name__=='__main__':
    args = docopt(__doc__+dataio.get_flags_doc(), argv=sys.argv[1:])
    
    gpuid  = int(args['--gpu'])
    mbsize = int(args['--mb'])
    epoch  = int(args['--epoch'])
    lr = float(args['--lr'])
    mm = float(args['--mm'])
    re = float(args['--re'])
    seed = int(args['--seed'])
    rbmtype = args['--rt']
    visnum = int(args['<visnum>'])
    hidnum = int(args['<hidnum>'])

    if rbmtype!="gb" and rbmtype!="bb":
        util.stderr("Unknown RBM type: %s" % rbmtype)
    elif rbmtype == "gb":
        bbrbm = False
    else:
        bbrbm = True

    data = dataio.dataio(args['<file>'], args['--df'], visnum).astype(np.float32)
    ndata = data.shape[0]
    
    if gpuid < 0:
        trainer = train_cpu(visnum, hidnum, seed, bb=bbrbm)
    else:
        trainer = train_gpu(visnum, hidnum, seed, bb=bbrbm, gpuid=gpuid)
        np.random.seed(seed)

    for i in range(epoch):
        e = 0.0
        mblst = np.random.permutation(ndata)

        for mb in range(0, ndata, mbsize):
            e += trainer.train(data[mblst[mb:mb+mbsize]], lr, re, mm)
        e /= (ndata/mbsize)
        util.stdout("%4d th-epoch mse= %9e\n" % (i, e))
