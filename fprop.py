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
   --ot=<f4le|f4be|f4ne|text>    output type is little/big/native endian 4-byte float or text format.
   --df=<str>    Sample data format flags.The flag's detail is follow.
   --mbsize=<num>  Size of mini-batch [default: 512].
""" 

import sys, time
from docopt import docopt

import numpy as np
from chainer import cuda, function, Variable
import chainer.functions as F

import util,dataio

def savebin(array, of):
    array.tofile(of)
def savetxt(array, of):
    np.savetxt(of, array, fmt="%+.9e")

if __name__=='__main__':
    args = docopt(__doc__, argv=sys.argv[1:])
    gpu = not bool(args['--cpu']) 
    nnf = args['--nn']
    otype = args['--ot']
    of  = args['--of']
    df  = args['--df']
    mbsize = int(args['--mbsize'])
    
    model, actfs = dataio.loadnn(nnf)
    nlayer = len(actfs)
    data = dataio.dataio(args['<file>'], df, model.l_0.W.shape[1]).astype(np.float32)
    
    if gpu:
        cuda.init()
        model.to_gpu()
        
    if otype == 'f4be': 
        be = True
        save = savebin
    elif otype == 'f4ne' or otype == 'f4le': 
        be = False
        save = savebin
    elif otype == 'text':
        be = False
        save = savetxt
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
        save(y, f)
    f.close()

