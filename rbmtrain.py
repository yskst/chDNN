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
from chainer import cuda, function, Variable
import chainer.functions as F

import util,dataio



def _create_empty_like(src):
    if isinstance(src, cuda.GPUArray):
        return cuda.empty_like(src)
    else:
        return np.empty_like(src)

_cudot = cuda.culinalg.dot
_cusum = cuda.cumisc.sum


class bbRBM(function.Function):
    """ The Gaussian-Bernoulli Restricted Boltzman Machine(GBRBM) """
    def __init__(self, vis_size, hid_size, act_func=F.sigmoid,
                       init_w=None, init_hbias=None, init_vbias=None, 
                       seed=1234, wscale=0.01):
        self.W      = None
        self.gW    = None
        self.hbias  = None
        self.ghbias = None
        self.vbias  = None
        self.gvbias = None
        
        self.seed = seed
        self.f = act_func
        np.random.seed(seed)

        if init_w is not None:
            assert init_w.shape == (vis_size, hid_size)
            self.W = init_w
        else:
            self.W = np.random.normal(0, wscale, 
                                    (vis_size, hid_size)).astype(np.float32)

        if init_hbias is not None:
            assert init_hbias.shape == (hid_size,)
            self.hbias = init_hbias
        else:
            self.hbias = np.zeros(hid_size, dtype=np.float32)
      
        if init_vbias is not None:
            assert init_vbias.shape == (vis_size,)
            self.vbias = init_vbias
        else:
            self.vbias = np.zeros(vis_size, dtype=np.float32)

        self.gW     = _create_empty_like(self.W)
        self.ghbias = _create_empty_like(self.hbias)
        self.gvbias = _create_empty_like(self.vbias)
        self.mse = None


    _linear_bias_gpu = cuda.elementwise(
            'float *x, float *b, int ndim',
            'x[i] += b[i % ndim]',
            'linear_bias')

    def parameter_names(self):
        return 'W', 'vbias', 'hbias'
    def gradient_names(self):
        return 'gW', 'gvbias', 'ghbias'

    def _linear_cpu(self, x, w, bias):
        return x.dot(w) + bias
    def _linear_gpu(self, x, w, bias):
        with cuda.using_cumisc():
            y = cuda.culinalg.dot(x, w)
        self._linear_bias_gpu(y, bias, bias.size)
        return y

    
    def _reconst_cpu(self, h):
        return self.f(self._linear_cpu(h, self.W.T.copy(), self.vbias))
    def _reconst_gpu(self, h):
        return self.f(self._linear_gpu(h, self.W.T.copy(), self.vbias))

    def forward_cpu(self, x):
        h0act = self.f(Variable(self._linear_cpu(x, self.W, self.hbias)))
        h0act = h0act.data
        h0smp = np.random.binomial(1, h0act, h0act.shape).astype(np.float32)
        v1act = self._reconst_cpu(h0smp)
        h1act = self.f(Variable(self._linear_cpu(v1act, self.W, self.hbias)))
        h1act = h1act.data

        self.mse = np.mean((x-v1act)**2)
        return h0act, v1act, h1act

    def forward_gpu(self, x):
        h0act = self.f(Variable( self._linear_gpu(x, self.W, self.hbias)))
        h0act = h0act.data
        h0smp = h0act < self.randgen.gen_uniform(h0act.shape, np.float32)
        v1act = self._reconst_gpu(h0smp)
        h1act = self.f(Variable(self._linear_gpu(v1act, self.W, self.hbias)))
        h1act = h1act.data
        
        self.mse = cuda.cumisc.mean((x-v1act)**2)
        return h0act, v1act, h1act

    def backward_cpu(self, x):
        ndata = x.shape[0]
        h0act, v1act, h1act = self.forward_cpu(x)
        gW     = (np.dot(v1act.T, h1act) - np.dot(x.T, h0act))    /ndata
        ghbias = (np.sum(h1act, axis=0) - np .sum(h0act, axis=0)) /ndata 
        gvbias = (np.sum(v1act, axis=0) - np.sum(x, axis=0))      /ndata
        return gW, ghbias, gvbias

    def backward_gpu(self, x):
        ndata = x.shape[0]
        h0act, v1act, h1act = self.forward_gpu(x)
        gW = (_cudot(v1act,h1act, transa='T') - 
                               _cudot(x, h0act, transa='T')) / ndata
        ghbias = (_cusum(h1act, axis=0) - _cusum(h0act,axis=0)) / ndata
        gvbias = (_cusum(v1act, axis=0) - _cusum(x, axis=0))    / ndata
        return gW, ghbias, gvbias

    def train_cpu(self, x, lr, mm, re):
        gW, ghbias, gvbias = self.backward_cpu(x)

        self.gW = -lr*gW + mm*self.gW -re*self.W
        self.ghbias = -lr*ghbias +mm*self.ghbias
        self.gvbias = -lr*gvbias +mm*self.gvbias
        
        # MSE is alucurated in forward_cpu called from backward_cpu
        return self.mse 

    def train_gpu(self, x, lr, mm, re):
        x_gpu = cuda.to_gpu(x)
        gW, ghbias, gvbias = self.backward_gpu(x_gpu)

        self.gW = -lr*gW + mm*self.gW -re*self.W
        self.ghbias = -lr*ghbias +mm*self.ghbias
        self.gvbias = -lr*gvbias +mm*self.gvbias
        
        # MSE is alucurated in forward_gpu called from backward_gpu
        return cuda.to_cpu (self.mse) 


    def to_gpu(self, device=None):
        cuda.seed(self.seed)
        self.randgen = cuda.get_generator(device)
        super(bbRBM, self).to_gpu(device)

class gbRBM(bbRBM):
   def _reconst_cpu(self, h):
       return self._linear_cpu(h, self.W.T.copy(), self.vbias)
   def _reconst_gpu(self, h):
       return self._linear_gpu(h, self.W.T.copy(), self.vbias)



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
        rbm = gbRBM(visnum, hidnum)
    else:
        rbm = bbRBM(visnum, hidnum)

    data = dataio.dataio(args['<file>'], args['--df'], visnum).astype(np.float32)
    ndata = data.shape[0]
    
    if gpuid < 0:
        trainer = rbm.train_cpu
    else:
        trainer = rbm.train_gpu
        cuda.init()
        rbm.to_gpu()

    for i in range(epoch):
        e = 0.0
        mblst = np.random.permutation(ndata)

        for mb in range(0, ndata, mbsize):
            e += trainer(data[mblst[mb:mb+mbsize]], lr, re, mm)
        e /= (ndata/mbsize)
        util.stdout("%4d th-epoch mse= %9e\n" % (i, e))
