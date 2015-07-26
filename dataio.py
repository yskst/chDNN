#!/usr/bin/python
""" Load data from File. """

import sys
import numpy as np

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

def saveRBM(f, rbm):
    np.savez(f, W_0=rbm.W, bias_0=rbm.hbias, vbias_0=rbm.vbias, f_0=rbm.f.__name__)


if __name__=='__main__':
    if len(sys.argv) < 2:
        stdout("Usage: %s [flag] [files]..\n\n")
        stdout(dataio.__doc__+"\n")
        
    fmt_flag = sys.argv[1]
    for fname in sys.argv[2:]:
        d = dataio(fname, fmt_flag)
        np.savetxt(sys.stdout, d, fmt='%+.6e')

