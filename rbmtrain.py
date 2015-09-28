#!/usr/bin/python

"""Training RBM.
Usage: 
  rbmtrain.py [options] <visnum> <hidnum> <file>
  rbmtrain.py -h | --help
options:
   -h, --help    Show this help.
   --cpu         Run on CPU
   --of=<file>   output file name.(npz file)
   --df=<str>    sample data format flags.The flag's detail is follow.
   --mb=<num>    mini-batch size.
   -e <num>, --epoch=<num>   the number of ephoch.
   --lr <val>    learning rate [default: 0] 
   --mm <val>    momentum [default: 0] 
   --re <val>    regulalizer. [default: 0]
   --rt <bb|rb>  bb=bernoulli-bernoulli, gb=gaussian-bernoulli.
   --af <str>    Activate function.[default: Sigmoid]
   --seed <NUM>  The seed of random value. [default: 1234]
""" 

import sys, time
from docopt import docopt

import numpy as np
from chainer import cuda, function, Variable
import chainer.cuda.cupy as cupy
import chainer.functions as F

import util,dataio



def _empty_like(src):
    if isinstance(src, cuda.GPUArray):
        return cuda.empty_like(src)
    else:
        return np.empty_like(src)

_cudot = cupy.dot
_cusum = cupy.sum


class bbRBM(function.Function):
    """ The Gaussian-Bernoulli Restricted Boltzman Machine(GBRBM) """
    def __init__(self, vis_size, hid_size, act_func=F.sigmoid,
                       init_w=None, init_hbias=None, init_vbias=None, 
                       seed=1234, wscale=1e-1):
        self.W      = None
        self.gW    = None
        self.hbias  = None
        self.ghbias = None
        self.vbias  = None
        self.gvbias = None
        
        self.seed = seed
        self.f = act_func
        
        if init_w is not None:
            assert init_w.shape == (vis_size, hid_size)
            self.W = init_w
        else:
            """self.W = np.random.RandomState(seed).uniform(
                        low=-4.0*np.sqrt(6.0/(vis_size+hid_size)),
                        high=4.0*np.sqrt(6.0/(vis_size+hid_size)),
                        size=(vis_size, hid_size)).astype(np.float32)"""
            self.W = np.random.RandomState(seed).normal(0, wscale,(vis_size, hid_size)).astype(np.float32)

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

        self.gW     = _empty_like(self.W)
        self.ghbias = _empty_like(self.hbias)
        self.gvbias = _empty_like(self.vbias)
        self.mse = None

    def init_grads(self):
        self.gW.fill(0)
        self.ghbias.fill(0)
        self.gvbias.fill(0)
    

    _linear_bias_gpu = cuda.elementwise(
            'float *x, float *b, int ndim',
            'x[i] += b[i % ndim]',
            'linear_bias')

    def parameter_names(self):
        return 'W', 'vbias', 'hbias'
    def gradient_names(self):
        return 'gW', 'gvbias', 'ghbias'

    def _linear_cpu(self, x, w, bias, transw='N'):
        if transw=='N':
            return x.dot(w) + bias
        else:
            return x.dot(w.T) + bias
    def _linear_gpu(self, x, w, bias, transw='N'):
        y = _cudot(x, w.transpose())
        self._linear_bias_gpu(y, bias, bias.size)
        return y

    
    def _reconst_cpu(self, h):
        return self.f(Variable(self._linear_cpu(h, self.W, self.vbias, 'T'))).data
    def _reconst_gpu(self, h):
        return self.f(Variable(self._linear_gpu(h, self.W, self.vbias, 'T'))).data

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
        h0smp = h0act > cupy.random.rand(h0act.shape, np.float32)
        v1act = self._reconst_gpu(h0smp)
        h1act = self.f(Variable(self._linear_gpu(v1act, self.W, self.hbias)))
        h1act = h1act.data
        
        self.mse = cupy.mean((x-v1act)**2)
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
        gW = (_cudot(v1act,h1act, transa='T') - _cudot(x, h0act, transa='T')) / ndata
        ghbias = (_cusum(h1act, axis=0) - _cusum(h0act,axis=0)) / ndata
        gvbias = (_cusum(v1act, axis=0) - _cusum(x, axis=0))    / ndata

        return gW, ghbias, gvbias

    def train_cpu(self, x, lr, mm, re):
        gW, ghbias, gvbias = self.backward_cpu(x)

        self.gW     = -lr*gW     +mm*self.gW     -re*self.W
        self.ghbias = -lr*ghbias +mm*self.ghbias
        self.gvbias = -lr*gvbias +mm*self.gvbias
        
        self.W += self.gW
        self.hbias += self.ghbias
        self.vbias += self.gvbias
        
        # MSE is calucurated in forward_cpu called from backward_cpu
        return self.mse 

    def train_gpu(self, x, lr, mm, re):
        x_gpu = cuda.to_gpu(x)
        gW, ghbias, gvbias = self.backward_gpu(x_gpu)
        
        self.gW     = -lr*gW     +mm*self.gW     -re*self.W
        self.ghbias = -lr*ghbias +mm*self.ghbias
        self.gvbias = -lr*gvbias +mm*self.gvbias
        
        self.W += self.gW
        self.hbias += self.ghbias
        self.vbias += self.gvbias
        # MSE is alucurated in forward_gpu called from backward_gpu
        return cuda.to_cpu (self.mse) 

    def to_gpu(self, device=None):
        self.randgen = cuda.get_generator(device)
        cupy.random.seed(self.seed)
        super(bbRBM, self).to_gpu(device)

class gbRBM(bbRBM):
    def _reconst_cpu(self, h):
        return self._linear_cpu(h, self.W.T.copy(), self.vbias)
    def _reconst_gpu(self, h):
        return self._linear_gpu(h, self.W, self.vbias, 'T')




if __name__=='__main__':
    args = docopt(__doc__+dataio.get_flags_doc(), argv=sys.argv[1:])
    
    outf    = args['--of']
    rbmtype = args['--rt']
    gpu  = not args['--cpu']
    mbsize = int(args['--mb'])
    epoch  = int(args['--epoch'])
    lr = float(args['--lr'])
    mm = float(args['--mm'])
    re = float(args['--re'])
    actf = args['--af']
    seed = int(args['--seed'])
    visnum = int(args['<visnum>'])
    hidnum = int(args['<hidnum>'])

    if rbmtype!="gb" and rbmtype!="bb":
        util.stderr("Unknown RBM type: %s" % rbmtype)
    elif rbmtype == "gb":
        rbm = gbRBM(visnum, hidnum, seed=seed)
    else:
        rbm = bbRBM(visnum, hidnum)
    
    af = dataio.str2actf(actf)

    data = dataio.dataio(args['<file>'], args['--df'], visnum).astype(np.float32)
    ndata = data.shape[0]
    
    if not gpu:
        trainer = rbm.train_cpu
        xp = np
    else:
        trainer = rbm.train_gpu
        xp = cupy
        cuda.check_cuda_available()
        cuda.get_device(0).use()
        rbm.to_gpu()

    rbm.init_grads()
    np.random.seed(seed)
    mbnum = ndata / mbsize
    for i in range(epoch):
        t1 = time.clock()
        e = 0.0
        mblst = np.random.permutation(ndata)
        for mb in range(0, ndata, mbsize):
            e += trainer(data[mblst[mb:mb+mbsize]], lr, mm, re)
        e /= mbnum
        t2 = time.clock()
        util.stdout("%4d th-epoch mse= %9e (%f sec)\n" % (i, e, t2-t1))
        sys.stdout.flush()
    
    if gpu:
        rbm.to_cpu()
    dataio.saveRBM(outf, af, rbm.W.T.copy(), rbm.hbias, rbm.vbias)
