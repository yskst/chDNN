#!/usr/bin/python

"""Forward Propagation.
Usage: 
  fprop.py [options] <file>
  fprop.py | --help
options:
   -h, --help    Show this help.
   --cpu         Run on CPU.
   --nn=<file>   Parameter file of Neural Network or RBM
   --of=<file>   Output file name.
   --ot=<f4le|f4be|f4ne>    output type is little/big/native endian 4-byte float.
   --df=<str>    Sample data format flags.The flag's detail is follow.
   --mbsize=<num>  Size of mini-batch [default: 256].
""" 

import sys, time
from docopt import docopt

import numpy as np
from chainer import cuda, function, Variable
import chainer.functions as F

import util,dataio


if __name__=='__main__':
    args = docopt(__doc__, argv=sys.argv[1:])
    gpu = not bool(args['--cpu']) 
    nnf = args['--nn']
    endian = args['--ot']
    of  = args['--of']
    df  = args['--df']
    mbsize = int(args['--mbsize'])
    
    model, actfs = dataio.loadnn(nnf)
    nlayer = len(actfs)
    data = dataio.dataio(args['<file>'], df, model.l_0.W.shape[1])
    
    if gpu:
        cuda.init()
        model.to_gpu()
        
    if endian == 'f4be': be = True
    elif endian == 'f4ne' or endian == 'f4le': be = False
    else:
        util.panic("Unknown endian type: " + endian)
    
    def forward(x_data):
        x = Variable(x_data)
        h = actfs[0](model.l_0(x))
        for i in range(1, nlayer):
            l = getattr(model, 'l_'+str(i))
            h = actfs[i](l(h))
        return h
    
    ndata = data.shape[0]
    f = open(of, 'wb')
    for i in range(0, ndata, mbsize):
        x_batch = data[i:i+mbsize]
        if gpu:
            x_batch = cuda.to_gpu(x_batch)
        y = forward(x_batch).data
        if gpu:
            y = cuda.to_cpu(y)
        y.byteswap(be)
        y.tofile(f)
    
    nbatch = ndata/mbsize
    x_batch = data[nbatch*mbsize:]
    if gpu:
        x_batch = cuda.to_gpu(x_batch)
    y = forward(x_batch).data
    if gpu:
        y = cuda.to_cpu(y)
    y.byteswap(be)
    y.tofile(f)

    f.close()

