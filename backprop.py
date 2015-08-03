#!/usr/bin/python

"""Training RBM.
Usage: 
  trainrbm.py [options] <file> <tar>
  trainrbm.py -h | --help
options:
   -h, --help    Show this help.
   --cpu         Process on the CPU.
   --mlp=<file>  The initial values of MLP parameter.
   --of=<file>   output file name.(npz file)
   --df=<str>    sample data format flags.The flag's detail is follow.
   --tf=<str>    the target data format flags.
   --tt=<c|f>    The target type is category or feature. 
   --mb=<num>    mini-batch size.
   -e <num> --epoch=<num>   the number of ephoch.
   --lr <val>    learning rate [default: 0] 
   --mm <val>    momentum [default: 0] 
   --re <val>    regulalizer. [default: 0]
   ---dr <val>   The probability of dropout. If 0, do not applying dropout. [default: 0]
   --seed <NUM>  The seed of random value. [default: 1234]
""" 

import sys, time
from docopt import docopt

import numpy as np
from chainer import cuda, function, Variable, FunctionSet, optimizers
import chainer.functions as F

import util,dataio

def _str2actf(s):
    sl = str(s).lower()
    if   sl == "sigmoid" or sl == "sigmoidlayer": return F.sigmoid
    elif sl == "softmax" or sl == "softmaxlayer": return F.softmax
    elif sl == "relu":    return F.relu
    elif sl == "tanh":    return F.tanh
    else:
        util.panic("Unknown activate function name %s \n" % s)

def _load_nn(fileobj):
    d = np.load(fileobj)
    i = 0
    actfs = []
    params = {}
    while 'w_'+str(i) in d.keys():
        s = str(i)
        w = d['w_'+s].T.copy()
        params['l'+s] = F.Linear(w.shape[1], w.shape[0], initialW=w, initial_bias=d['hbias_'+s])
        actf = _str2actf(d['type_'+s])
        actfs.append(actf)
        i+=1
    model = FunctionSet(**params)
    return model, actfs

def forward_mse(x_data, y_data, model, actf):
    x = Variable(x_data)
    y = Variable(y_data)
    h = actf[0](model.l0(x))
    for i in range(1, len(actf)):
        m = getattr(model, 'l'+str(i))
        h = actf[i](m(h))
    e = mean_squared_error(h, y)
    return e,e

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
    args = docopt(__doc__+dataio.get_flags_doc(), argv=sys.argv[1:])
    dataform= args['--df']
    outf    = args['--of']
    tarform = args['--tf']
    ttype   = args['--tt']
    nn      = args['--mlp']
    gpu     = not bool(args['--cpu'])
    mbsize  = int(args['--mb'])
    epoch   = int(args['--epoch'])
    lr      = float(args['--lr'])
    mm      = float(args['--mm'])
    re      = float(args['--re'])
    seed    = int(args['--seed'])
    tarf    = args['<tar>']
    trainf  = args['<file>']

    model, actfs = _load_nn(nn)
    if gpu:
        cuda.init()
        model.to_gpu()
    
    optimizer = optimizers.MomentumSGD(lr=lr,momentum=mm)
    optimizer.setup(model.collect_parameters())
    
    nlayer = len(actfs)
    idim = model.l0.W.shape[1]
    odim = getattr(model, 'l'+str(nlayer-1)).W.shape[0]


    data = dataio.dataio(trainf, dataform, idim).astype(np.float32)
    
    if ttype == 'c':
        forward = forward_cross_entoropy
        tar = dataio.dataio(tarf, tarform).astype(np.int32)
    elif ttype == 'f':
        forward = forward_mse
        tar = data.dataio(tarf, tarform, 0).astype(np.float32)

    ndata = data.shape[0]
    nmb = ndata / mbsize
    np.random.seed(seed)
    for ep in range(epoch):
        mse = 0.0
        mean_acc = 0.0
        mb = np.random.permutation(ndata)
        for i in range(0, ndata, mbsize):
            x_batch = data[mb[i:i+mbsize]]
            y_batch = tar[mb[i:i+mbsize]]

            if gpu:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            optimizer.zero_grads()
            loss, acc = forward(x_batch, y_batch, model, actfs)
            loss.backward()
            optimizer.update()
            mse += loss.data
            mean_acc += acc.data
        mse = cuda.to_cpu(mse)
        mean_acc /= cuda.to_cpu(mse)
        mse /= nmb
        mean_acc /= nmb
        print ep, mse, mean_acc
        #util.stdout('%4d th, mse= %.8e mean_of_acc= %.8e\n'% (i, mse, mean_acc))
        sys.stdout.flush()

