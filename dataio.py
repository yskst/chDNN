#!/usr/bin/python
""" Load data from File. """

import sys
import numpy as np

from chainer import FunctionSet
import chainer.functions as F


import util


def __parse_flags__(fmt):
    """ parse file format flags """
    r = ""
    e = fmt[2:]
    if   e == "le": r = "<"
    elif e == "be": r = ">"
    elif e == "ne": r = "="
    else:
        raise NameError("The encoding %s is unknown." % fmt)
    return r + fmt[0:2]

def str2actf(s):
    sl = str(s).lower()
    if   sl == "sigmoid" or sl == "sigmoidlayer": return F.sigmoid
    elif sl == "softmax" or sl == "softmaxlayer": return F.softmax
    elif sl == "relu"    or sl == "relulayer":  return F.relu
    elif sl == "tanh"    or sl == "tanhlayer": return F.tanh
    else:
        util.panic("Unknown activate function name %s \n" % s)

def get_flags_doc():
    s ="""    
    The format flags is three type. If the file is ascii text format, the flag is 'text'. If the file is numpy binary format, the flag is 'npy'.
    If the file is binary format, the flag is combination of encoding and endian(e.g. f4le means little nedian 4byte float).

    The encoding is concatecated 2 characters. The first one is the kind of data. The second one is the number of byte per one sample. 
    Available kind is follow:
        i: int
        u: unsigned int
        f: float
   The endian is follow:
        le: little endian
        be: big endian
        ne: native endian
"""
    return s


def dataio(f, fmt, ndim=None):
    """ Load data from file.
    parameter f:    File like object or filename.
    parameter fmt:  File format flags. Details can get `get_flags_doc()` function.
    parameter ncol: The number of column to reshape. If False or None, Nothing to do.
    """
    if fmt == "npy":
        return np.load(f)
    elif fmt == "text":
        return np.loadtxt(f)
    else:
        dtype = __parse_flags__(fmt)
        m = np.fromfile(f, dtype)
        if ndim:
            return np.reshape(m, (-1, ndim))
        return m

def saveRBM(f, func, w, hbias, vbias=None):
    """ Save RBM parameter. 
        Prameters:
            f:     File like object to save
            func:  Activate function.
            w:     Weight parameter.(outsize, insize)
            hbias: Bias parameter of hidden layer.
            vbias: Bias paramter of visible layer.
    """
    assert w.shape[0] == hbias.shape[0]
    d = {'type_0':func.__name__, 'w_0':w, 'hbias_0':hbias}
    if vbias is not None:
        d['vbias_0'] = vbias
    np.savez(f, **d)

def savenn(f, model, actf):
    nlayer = len(actf)
    d = {}
    for i in range(nlayer):
        s = str(i)
        l = getattr(model, 'l_'+s)
        d['w_'+s] = l.W
        d['hbias_'+s] = l.b
        d['type_'+s] = actf[i].__name__
    np.savez(f, **d)

def loadnn(f):
    """ Load RBM or Neural Network prameter from file like object. """
    d = np.load(f)
    params = {}
    actfs = []
    i = 0
    while 'type_'+str(i) in d.keys():
        s = str(i)
        w = d['w_'+s]
        params['l_'+s] = F.Linear(w.shape[1], w.shape[0], 
                            initialW=w, initial_bias=d['hbias_'+s])
        actfs.append(str2actf(d['type_'+s]) )
        i+=1
    model = FunctionSet(**params)
    return model, actfs


if __name__=='__main__':
    if len(sys.argv) < 2:
        stdout("Usage: %s [flag] [files]..\n\n")
        stdout(dataio.__doc__+"\n")
        
    fmt_flag = sys.argv[1]
    for fname in sys.argv[2:]:
        d = dataio(fname, fmt_flag)
        np.savetxt(sys.stdout, d, fmt='%+.6e')

