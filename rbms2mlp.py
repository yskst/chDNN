#!/usr/bin/python

"""Convert from  RBMs to MLP.
Usage: 
  trainrbm.py [options] <files>...
  trainrbm.py -h | --help
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
    files = args["files"]
    d = {}
    for i, f in enumerate(files):
        s = str(i)
        rbm = np.load(f)
        d['w_'+s]     = rbm['w_0']
        d['hbias_'+s] = rbm['hbias_0']
        d['type_'+s]  = rbm['type_0']
    np.savez(f, **d)
