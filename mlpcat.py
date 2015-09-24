#!/usr/bin/python

"""Concatenate some MLPs..
Usage: 
  rbms2mlp.py [options] <files>...
  rbms2mlp.py -h | --help
options:
   -h, --help    Show this help.
   --of=<file>   output file name.(npz file)
""" 

from docopt import docopt

import numpy as np
import util,dataio


if __name__=='__main__':
    args = docopt(__doc__)
    of = args["--of"]
    files = args["<files>"]
    d = {}
    l = 0
    for f in files:
        mlp = np.load(f)
        i = 0
        while 'type_'+ str(i) in mlp.keys():
            si = str(i)
            sl = str(l)
            d['type_'+sl] = mlp['type_'+si]
            d['w_'+sl] = mlp['w_'+si]
            d['hbias_'+sl] = mlp['hbias_'+si]
            i += 1
            l += 1

    np.savez(of, **d)
