#!/usr/bin/python

"""Training RBM.
Usage: 
  backprop.py [options] <file> <tar>
  backprop.py -h | --help
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
   --dr <val>   The probability of dropout. If 0, do not applying dropout. [default: 0]
   --seed <NUM>  The seed of random value. [default: 1234]
""" 

import sys, time
from docopt import docopt

import numpy as np
from chainer import cuda, function, Variable, FunctionSet, optimizers
import chainer.functions as F

import util,dataio

def forward_mse(x_data, y_data, model, actf, dr=False):
    x = Variable(x_data)
    y = Variable(y_data)
    h = actf[0](model.l_0(x))
    use_dr = False
    if dr and dr != 0.0:
        use_dr = True

    for i in range(1, len(actf)):
        m = getattr(model, 'l_'+str(i))
        h = F.dropout(actf[i](m(h)), ratio=dr, train=use_dr)
    e = F.mean_squared_error(h, y)
    return e,e


def forward_cross_entoropy(x_data, y_data, model, actf, dr=False):
    x = Variable(x_data)
    y = Variable(y_data)
    h = actf[0](model.l_0(x))
    
    use_dr=False
    if dr and dr != 0.0:
        use_dr=True

    l = len(actf)
    for i in range(1, l):
        m = getattr(model, 'l_'+str(i))
        if i < l-1:
            h = F.dropout(actf[i](m(h)), ratio=dr, train=use_dr)
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
    dr      = float(args['--dr'])
    seed    = int(args['--seed'])
    tarf    = args['<tar>']
    trainf  = args['<file>']

    model, actfs = dataio.loadnn(nn)
    if gpu:
        cuda.init()
        model.to_gpu()
    
    optimizer = optimizers.MomentumSGD(lr=lr,momentum=mm)
    optimizer.setup(model.collect_parameters())
    
    nlayer = len(actfs)
    idim = model.l_0.W.shape[1]
    odim = getattr(model, 'l_'+str(nlayer-1)).W.shape[0]


    data = dataio.dataio(trainf, dataform, idim).astype(np.float32)
    
    if ttype == 'c':
        forward = forward_cross_entoropy
        tar = dataio.dataio(tarf, tarform).astype(np.int32)
    elif ttype == 'f':
        forward = forward_mse
        tar = dataio.dataio(tarf, tarform, odim).astype(np.float32)
    
    assert data.shape[0] == tar.shape[0]
    ndata = data.shape[0]
    nmb = ndata / mbsize
    np.random.seed(seed)
    for ep in range(epoch):
        mse = 0.0
        mean_acc = 0.0
        mb = np.random.permutation(ndata)
        t1 = time.clock()
        for i in range(0, ndata, mbsize):
            x_batch = data[mb[i:i+mbsize]]
            y_batch = tar[mb[i:i+mbsize]]

            if gpu:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            optimizer.zero_grads()
            loss, acc = forward(x_batch, y_batch, model, actfs, dr)
            loss.backward()
            optimizer.update()
            optimizer.weight_decay(re) # L2 reguralization.
            mse      += float(cuda.to_cpu(loss.data))
            mean_acc += float(cuda.to_cpu(acc.data))
        mse /= nmb
        mean_acc /= nmb
        t2 = time.clock()
        util.stdout("%4d %s %s ( %f sec)\n" %(ep, str(mse), str(mean_acc), t2-t1))
        sys.stdout.flush()
    if gpu:
        model.to_cpu()
    dataio.savenn(outf,model, actfs)

