#!/usr/bin/python

"""Pick up some of MLP parameter.
Usage: 
  pick.py [options] <file>
  pick.py -h | --help
options:
   -h, --help    Show this help.
   --of=<file>   Output file.
   -p <0..>, --pick=<0..>   The index of layer to pick.(-p I to pick one, -p I:J to pick I to J)
""" 

import re

from docopt import docopt

import numpy as np
import util,dataio


if __name__=='__main__':
    args = docopt(__doc__)
    of = args["--of"]
    pick = args["--pick"]
    file = args["<file>"]
    
    nn, actfs = dataio.loadnn(file)

    p = []
    if re.match(r"^\d$", pick):
        p = [ int(pick) ]
    elif re.match(r"^\d:\d$", pick):
        r = s.rstrip().split(':')
        assert len(r) == 2
        r = map(int, r)
        p = range(r[0], r[1])
    else:
        util.panic("Pick up argument is Unknown format: %s" % pick)

    d = {}
    j = 0
    for i in p:
        l = getattr(nn, 'l_' + str(i))
        s = str(j)
        d['w_' + s] = l.W
        d['hbias_' + s] = l.b
        d['type_' + s]  = actfs[i].__name__
        j += 1
    np.save(of, **d)
