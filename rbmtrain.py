#!/usr/bin/python

# Training RBM.

import sys, fileinput
import util,fload

def print_help():
    stdout("""
Training RBM.
Usage: %s [options] [files]...
required options
---------------
    --of     [FILE]:  output file name.(npz file)
    --df     [str]:   sample format flags.The flag's detail is follow.
    --mb     [NUM]:   mini-batch size.
    -e, --ep [NUM]:   the number of ephoch.
    --lr, --mm --re [NUM]: learning rate, momentum, regulalizer.
                            [default=0]
    --rt     [bb|rb]: bb=bernoulli-bernoulli, gb=gaussian-bernoulli.

optional option
---------------
    --seed [NUM]:     The seed of random value.[default=1234]
    -h, --help:       Show this help.

data format
-----------
""")
    stdout(fload.get_flags_doc())

if __name__=='__main__':
    lr = 0
    mm = 0
    re = 0
    mbsize = None
    epoch = None
    ofile = None
    dtype = None
    rbmtype = None



    # option analysis
    args=[]
